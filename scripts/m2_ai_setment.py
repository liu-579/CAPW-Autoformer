#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
模块2：多维情感评分脚本
使用GPT对旅游评论进行多维度情感打分
"""

import json
import asyncio
import time
import logging
import sys
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Optional
from dataclasses import dataclass, asdict

from openai import AsyncOpenAI
from tqdm.asyncio import tqdm

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import m2_config


@dataclass
class SentimentScore:
    """情感评分结果"""
    id: str
    timestamp: str
    scores: Dict[str, float]
    success: bool
    error: Optional[str] = None


@dataclass
class CostStats:
    """成本统计"""
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    total_cost_usd: float = 0.0

    def add_usage(self, input_tokens: int, output_tokens: int, input_price: float, output_price: float):
        """添加一次请求的用量"""
        self.total_input_tokens += input_tokens
        self.total_output_tokens += output_tokens
        self.total_cost_usd += (input_tokens / 1_000_000 * input_price +
                                output_tokens / 1_000_000 * output_price)


class SentimentScorer:
    """情感评分器"""

    def __init__(self):
        """初始化评分器"""
        self.config = m2_config.get_config()
        self._setup_logging()
        self._setup_openai_client()
        self.cost_stats = CostStats()
        self.semaphore = asyncio.Semaphore(self.config['concurrency']['max_workers'])
        self.rate_limiter = RateLimiter(self.config['concurrency']['rate_limit'])

    def _setup_logging(self):
        """设置日志"""
        log_level = getattr(logging, self.config['logging']['level'])

        # 配置日志格式
        log_format = '[%(asctime)s] [%(levelname)s] %(message)s'
        date_format = '%Y-%m-%d %H:%M:%S'

        # 基本配置
        logging.basicConfig(
            level=log_level,
            format=log_format,
            datefmt=date_format
        )

        self.logger = logging.getLogger('SentimentScorer')

        # 如果需要保存日志
        if self.config['logging']['save_log']:
            log_file = Path(self.config['logging']['log_file'])
            log_file.parent.mkdir(parents=True, exist_ok=True)

            file_handler = logging.FileHandler(log_file, encoding='utf-8')
            file_handler.setFormatter(logging.Formatter(log_format, date_format))
            self.logger.addHandler(file_handler)

    def _setup_openai_client(self):
        """设置OpenAI客户端"""
        openai_config = self.config['openai']

        client_kwargs = {
            'api_key': openai_config['api_key']
        }

        if openai_config.get('base_url'):
            client_kwargs['base_url'] = openai_config['base_url']

        self.client = AsyncOpenAI(**client_kwargs)
        self.logger.info(f"OpenAI客户端初始化完成，使用模型: {openai_config['model']}")

    def _build_user_prompt(self, review: str) -> str:
        """构建用户提示词"""
        # 构建维度说明
        dimensions_text = "\n".join([
            f"- {dim['name']}（{dim['key']}）: {dim['description']}"
            for dim in self.config['dimensions']
        ])

        # 替换模板中的占位符
        prompt = self.config['user_prompt_template']
        prompt = prompt.replace('{review}', review)
        prompt = prompt.replace('{dimensions}', dimensions_text)

        return prompt

    async def _call_openai_with_retry(self, review: str, review_id: str) -> Optional[Dict[str, float]]:
        """带重试的OpenAI API调用"""
        retry_config = self.config['retry']

        for attempt in range(retry_config['max_attempts']):
            try:
                # 限流
                await self.rate_limiter.acquire()

                # 构建消息
                messages = [
                    {"role": "system", "content": self.config['system_prompt']},
                    {"role": "user", "content": self._build_user_prompt(review)}
                ]

                # 调用API
                response = await self.client.chat.completions.create(
                    model=self.config['openai']['model'],
                    messages=messages,
                    temperature=self.config['openai']['temperature'],
                    max_tokens=self.config['openai']['max_tokens'],
                    response_format={"type": "json_object"}
                )

                # 解析结果
                content = response.choices[0].message.content
                scores = json.loads(content)

                # 记录用量
                usage = response.usage
                self.cost_stats.add_usage(
                    usage.prompt_tokens,
                    usage.completion_tokens,
                    self.config['cost_estimation']['input_price'],
                    self.config['cost_estimation']['output_price']
                )
                self.cost_stats.successful_requests += 1

                # 验证返回的keys
                expected_keys = [dim['key'] for dim in self.config['dimensions']]
                if not all(key in scores for key in expected_keys):
                    raise ValueError(f"返回的维度不完整: {scores}")

                return scores

            except Exception as e:
                self.logger.warning(
                    f"请求失败 (尝试 {attempt + 1}/{retry_config['max_attempts']}): "
                    f"ID={review_id}, 错误={str(e)}"
                )

                if attempt < retry_config['max_attempts'] - 1:
                    # 计算延迟时间（指数退避）
                    delay = min(
                        retry_config['initial_delay'] * (retry_config['exponential_base'] ** attempt),
                        retry_config['max_delay']
                    )
                    await asyncio.sleep(delay)
                else:
                    self.cost_stats.failed_requests += 1
                    return None

        return None

    async def _score_single_review(self, item: dict) -> SentimentScore:
        """评分单条评论"""
        async with self.semaphore:
            review_id = item.get('id', 'unknown')
            review_text = item.get('text', '')

            self.cost_stats.total_requests += 1

            # 调用API
            scores = await self._call_openai_with_retry(review_text, review_id)

            # 处理结果
            if scores is None:
                # 失败后使用默认分数
                default_score = self.config['failure_handling']['default_score']
                scores = {dim['key']: default_score for dim in self.config['dimensions']}
                success = False
                error = "API调用失败，使用默认分数"
            else:
                success = True
                error = None

            return SentimentScore(
                id=review_id,
                timestamp=datetime.now().isoformat(),
                scores=scores,
                success=success,
                error=error
            )

    async def process_reviews(self, reviews: List[dict]) -> List[SentimentScore]:
        """批量处理评论"""
        self.logger.info(f"开始处理 {len(reviews)} 条评论")

        # 创建任务
        tasks = [self._score_single_review(item) for item in reviews]

        # 使用进度条执行
        if self.config['logging']['verbose']:
            results = await tqdm.gather(*tasks, desc="评分进度")
        else:
            results = await asyncio.gather(*tasks)

        return results

    def _save_results(self, results: List[SentimentScore]):
        """保存结果"""
        output_file = Path(self.config['output_file'])
        output_file.parent.mkdir(parents=True, exist_ok=True)

        # 转换为字典列表
        results_dict = [asdict(r) for r in results]

        # 保存JSON
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results_dict, f, ensure_ascii=False, indent=2)

        self.logger.info(f"✓ 结果已保存: {output_file}")

        # 保存失败记录
        if self.config['failure_handling']['save_failed_records']:
            failed_records = [r for r in results if not r.success]
            if failed_records:
                failed_file = Path(self.config['failure_handling']['failed_records_file'])
                failed_file.parent.mkdir(parents=True, exist_ok=True)

                with open(failed_file, 'w', encoding='utf-8') as f:
                    json.dump([asdict(r) for r in failed_records], f, ensure_ascii=False, indent=2)

                self.logger.warning(f"⚠ {len(failed_records)} 条失败记录已保存: {failed_file}")

    def _print_stats(self):
        """打印统计信息"""
        stats = self.cost_stats

        print("\n" + "=" * 60)
        print("评分统计报告")
        print("=" * 60)
        print(f"总请求数:     {stats.total_requests}")
        print(
            f"成功请求:     {stats.successful_requests} ({stats.successful_requests / max(stats.total_requests, 1) * 100:.1f}%)")
        print(f"失败请求:     {stats.failed_requests}")
        print(f"\n输入 Tokens:  {stats.total_input_tokens:,}")
        print(f"输出 Tokens:  {stats.total_output_tokens:,}")
        print(f"总 Tokens:    {stats.total_input_tokens + stats.total_output_tokens:,}")
        print(f"\n预估成本:     ${stats.total_cost_usd:.4f} USD")
        print(f"               ≈ ¥{stats.total_cost_usd * 7.2:.2f} CNY (按汇率7.2计算)")
        print("=" * 60)

    async def run(self):
        """运行评分流程"""
        start_time = time.time()

        try:
            # 读取输入文件
            input_file = Path(self.config['input_file'])
            if not input_file.exists():
                raise FileNotFoundError(f"输入文件不存在: {input_file}")

            self.logger.info(f"读取输入文件: {input_file}")

            reviews = []
            with open(input_file, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        reviews.append(json.loads(line))

            self.logger.info(f"成功读取 {len(reviews)} 条评论")

            # 测试模式
            if self.config['test_mode']['enabled']:
                sample_size = self.config['test_mode']['sample_size']
                reviews = reviews[:sample_size]
                self.logger.warning(f"⚠ 测试模式已启用，只处理前 {sample_size} 条数据")

            # 处理评论
            results = await self.process_reviews(reviews)

            # 保存结果
            self._save_results(results)

            # 打印统计
            self._print_stats()

            elapsed = time.time() - start_time
            self.logger.info(f"\n✓ 全部完成！耗时: {elapsed:.2f} 秒")

        except Exception as e:
            self.logger.error(f"✗ 运行失败: {str(e)}")
            raise


class RateLimiter:
    """速率限制器"""

    def __init__(self, max_per_minute: int):
        self.max_per_minute = max_per_minute
        self.tokens = max_per_minute
        self.updated_at = time.monotonic()
        self.lock = asyncio.Lock()

    async def acquire(self):
        """获取令牌"""
        async with self.lock:
            while self.tokens <= 0:
                # 计算需要等待的时间
                now = time.monotonic()
                elapsed = now - self.updated_at

                # 补充令牌
                self.tokens = min(
                    self.max_per_minute,
                    self.tokens + elapsed * self.max_per_minute / 60.0
                )
                self.updated_at = now

                if self.tokens <= 0:
                    await asyncio.sleep(0.1)

            self.tokens -= 1


async def main():
    """主函数"""
    scorer = SentimentScorer()
    await scorer.run()


if __name__ == "__main__":
    asyncio.run(main())