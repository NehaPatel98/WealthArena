"""
Comprehensive Metrics System for WealthArena Trading System

This module provides comprehensive metrics collection and reporting for:
- Backend API Metrics (Response Time, Error Rate, CPU, Memory, Disk I/O)
- Pipeline Metrics (DAG Execution, Task Failure, Duration Variability)
- Code Metrics (Cyclomatic Complexity, Duplication Rate, Technical Debt)
- Data Metrics (Query Latency, Index Usage, Disk Usage, etc.)
- Scraping Metrics (Success Rate, Blocked URLs, Throughput, etc.)
- ML Metrics (Inference Time, RMSE, AUC, F1, etc.)
- Testing Metrics (Coverage, Failure Rate, Bug Detection)
"""

import time
import psutil
import logging
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import json
import numpy as np
import pandas as pd
from pathlib import Path
import threading
from collections import defaultdict, deque
import requests
import subprocess
import os

logger = logging.getLogger(__name__)


@dataclass
class BackendAPIMetrics:
    """Backend API performance metrics"""
    response_time: float = 0.0
    error_rate: float = 0.0
    cpu_utilization: float = 0.0
    memory_utilization: float = 0.0
    disk_io: float = 0.0
    requests_per_second: float = 0.0
    active_connections: int = 0
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class PipelineMetrics:
    """Pipeline execution metrics"""
    dag_execution_success_rate: float = 0.0
    task_failure_rate: float = 0.0
    task_duration_variability: float = 0.0
    average_task_duration: float = 0.0
    total_tasks_executed: int = 0
    failed_tasks: int = 0
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class CodeMetrics:
    """Code quality metrics"""
    cyclomatic_complexity: float = 0.0
    duplication_rate: float = 0.0
    technical_debt: float = 0.0
    lines_of_code: int = 0
    test_coverage: float = 0.0
    code_smells: int = 0
    maintainability_index: float = 0.0
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class DataMetrics:
    """Data store performance metrics"""
    avg_query_latency: float = 0.0
    p95_query_latency: float = 0.0
    p99_query_latency: float = 0.0
    index_usage_percentage: float = 0.0
    disk_usage: float = 0.0
    cpu_usage: float = 0.0
    memory_usage: float = 0.0
    open_connections: int = 0
    error_rate: float = 0.0
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class ScrapingMetrics:
    """Web scraping performance metrics"""
    success_rate: float = 0.0
    blocked_urls_percentage: float = 0.0
    error_rate_by_type: Dict[str, float] = field(default_factory=dict)
    scraping_throughput: float = 0.0
    data_loss_rate: float = 0.0
    response_time: float = 0.0
    ip_blocks_frequency: float = 0.0
    pages_scraped_successfully: int = 0
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class MLMetrics:
    """Machine learning model metrics"""
    inference_time: float = 0.0
    rmse: float = 0.0
    auc: float = 0.0
    f1_score: float = 0.0
    precision: float = 0.0
    recall: float = 0.0
    accuracy: float = 0.0
    r2_score: float = 0.0
    model_size: float = 0.0
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class TestingMetrics:
    """Testing performance metrics"""
    test_coverage: float = 0.0
    failure_rate: float = 0.0
    bugs_detected_percentage: float = 0.0
    total_tests: int = 0
    passed_tests: int = 0
    failed_tests: int = 0
    test_duration: float = 0.0
    timestamp: datetime = field(default_factory=datetime.now)


class MetricsCollector:
    """Comprehensive metrics collector for WealthArena system"""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.metrics_history = {
            "backend_api": deque(maxlen=1000),
            "pipeline": deque(maxlen=1000),
            "code": deque(maxlen=1000),
            "data": deque(maxlen=1000),
            "scraping": deque(maxlen=1000),
            "ml": deque(maxlen=1000),
            "testing": deque(maxlen=1000)
        }
        
        # Performance tracking
        self.performance_data = defaultdict(list)
        self.start_time = time.time()
        
        # Threading for continuous collection
        self.collection_thread = None
        self.stop_collection = False
        
        logger.info("Metrics collector initialized")
    
    def start_continuous_collection(self, interval: int = 60):
        """Start continuous metrics collection"""
        
        if self.collection_thread and self.collection_thread.is_alive():
            logger.warning("Collection already running")
            return
        
        self.stop_collection = False
        self.collection_thread = threading.Thread(
            target=self._continuous_collection_loop,
            args=(interval,)
        )
        self.collection_thread.daemon = True
        self.collection_thread.start()
        
        logger.info(f"Started continuous metrics collection with {interval}s interval")
    
    def stop_continuous_collection(self):
        """Stop continuous metrics collection"""
        
        self.stop_collection = True
        if self.collection_thread:
            self.collection_thread.join(timeout=5)
        
        logger.info("Stopped continuous metrics collection")
    
    def _continuous_collection_loop(self, interval: int):
        """Continuous collection loop"""
        
        while not self.stop_collection:
            try:
                # Collect all metrics
                self.collect_backend_api_metrics()
                self.collect_pipeline_metrics()
                self.collect_code_metrics()
                self.collect_data_metrics()
                self.collect_scraping_metrics()
                self.collect_ml_metrics()
                self.collect_testing_metrics()
                
                time.sleep(interval)
                
            except Exception as e:
                logger.error(f"Error in continuous collection: {e}")
                time.sleep(interval)
    
    def collect_backend_api_metrics(self) -> BackendAPIMetrics:
        """Collect backend API metrics"""
        
        try:
            # System metrics
            cpu_percent = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            disk = psutil.disk_io_counters()
            
            # Network connections
            connections = len(psutil.net_connections())
            
            # Simulate API response time (in real implementation, this would be actual API calls)
            response_time = np.random.exponential(0.1)  # Simulated
            
            # Simulate error rate
            error_rate = np.random.beta(1, 99)  # Simulated
            
            metrics = BackendAPIMetrics(
                response_time=response_time,
                error_rate=error_rate,
                cpu_utilization=cpu_percent,
                memory_utilization=memory.percent,
                disk_io=disk.read_bytes + disk.write_bytes if disk else 0,
                requests_per_second=np.random.poisson(10),  # Simulated
                active_connections=connections
            )
            
            self.metrics_history["backend_api"].append(metrics)
            return metrics
            
        except Exception as e:
            logger.error(f"Error collecting backend API metrics: {e}")
            return BackendAPIMetrics()
    
    def collect_pipeline_metrics(self) -> PipelineMetrics:
        """Collect pipeline execution metrics"""
        
        try:
            # Simulate pipeline metrics (in real implementation, this would query actual pipeline data)
            success_rate = np.random.beta(95, 5)
            failure_rate = 1 - success_rate
            duration_variability = np.random.gamma(2, 0.5)
            avg_duration = np.random.exponential(30)
            
            metrics = PipelineMetrics(
                dag_execution_success_rate=success_rate,
                task_failure_rate=failure_rate,
                task_duration_variability=duration_variability,
                average_task_duration=avg_duration,
                total_tasks_executed=np.random.poisson(100),
                failed_tasks=int(failure_rate * 100)
            )
            
            self.metrics_history["pipeline"].append(metrics)
            return metrics
            
        except Exception as e:
            logger.error(f"Error collecting pipeline metrics: {e}")
            return PipelineMetrics()
    
    def collect_code_metrics(self) -> CodeMetrics:
        """Collect code quality metrics"""
        
        try:
            # Simulate code metrics (in real implementation, this would use SonarQube or similar)
            cyclomatic_complexity = np.random.gamma(2, 2)
            duplication_rate = np.random.beta(2, 98)
            technical_debt = np.random.gamma(3, 10)
            
            metrics = CodeMetrics(
                cyclomatic_complexity=cyclomatic_complexity,
                duplication_rate=duplication_rate,
                technical_debt=technical_debt,
                lines_of_code=np.random.poisson(10000),
                test_coverage=np.random.beta(80, 20),
                code_smells=np.random.poisson(50),
                maintainability_index=np.random.beta(70, 30)
            )
            
            self.metrics_history["code"].append(metrics)
            return metrics
            
        except Exception as e:
            logger.error(f"Error collecting code metrics: {e}")
            return CodeMetrics()
    
    def collect_data_metrics(self) -> DataMetrics:
        """Collect data store performance metrics"""
        
        try:
            # Simulate data store metrics
            avg_latency = np.random.exponential(0.05)
            p95_latency = avg_latency * np.random.gamma(2, 1.5)
            p99_latency = avg_latency * np.random.gamma(3, 2)
            
            metrics = DataMetrics(
                avg_query_latency=avg_latency,
                p95_query_latency=p95_latency,
                p99_query_latency=p99_latency,
                index_usage_percentage=np.random.beta(80, 20),
                disk_usage=np.random.beta(60, 40),
                cpu_usage=np.random.beta(30, 70),
                memory_usage=np.random.beta(40, 60),
                open_connections=np.random.poisson(20),
                error_rate=np.random.beta(1, 99)
            )
            
            self.metrics_history["data"].append(metrics)
            return metrics
            
        except Exception as e:
            logger.error(f"Error collecting data metrics: {e}")
            return DataMetrics()
    
    def collect_scraping_metrics(self) -> ScrapingMetrics:
        """Collect web scraping performance metrics"""
        
        try:
            # Simulate scraping metrics
            success_rate = np.random.beta(90, 10)
            blocked_rate = np.random.beta(5, 95)
            throughput = np.random.poisson(100)
            
            error_types = {
                "404": np.random.beta(2, 98),
                "timeout": np.random.beta(1, 99),
                "captcha": np.random.beta(3, 97),
                "rate_limit": np.random.beta(1, 99)
            }
            
            metrics = ScrapingMetrics(
                success_rate=success_rate,
                blocked_urls_percentage=blocked_rate,
                error_rate_by_type=error_types,
                scraping_throughput=throughput,
                data_loss_rate=np.random.beta(2, 98),
                response_time=np.random.exponential(0.5),
                ip_blocks_frequency=np.random.beta(1, 99),
                pages_scraped_successfully=np.random.poisson(1000)
            )
            
            self.metrics_history["scraping"].append(metrics)
            return metrics
            
        except Exception as e:
            logger.error(f"Error collecting scraping metrics: {e}")
            return ScrapingMetrics()
    
    def collect_ml_metrics(self) -> MLMetrics:
        """Collect machine learning model metrics"""
        
        try:
            # Simulate ML metrics
            inference_time = np.random.exponential(0.01)
            rmse = np.random.gamma(2, 0.1)
            auc = np.random.beta(80, 20)
            f1_score = np.random.beta(75, 25)
            
            metrics = MLMetrics(
                inference_time=inference_time,
                rmse=rmse,
                auc=auc,
                f1_score=f1_score,
                precision=np.random.beta(80, 20),
                recall=np.random.beta(75, 25),
                accuracy=np.random.beta(85, 15),
                r2_score=np.random.beta(70, 30),
                model_size=np.random.gamma(2, 10)
            )
            
            self.metrics_history["ml"].append(metrics)
            return metrics
            
        except Exception as e:
            logger.error(f"Error collecting ML metrics: {e}")
            return MLMetrics()
    
    def collect_testing_metrics(self) -> TestingMetrics:
        """Collect testing performance metrics"""
        
        try:
            # Simulate testing metrics
            coverage = np.random.beta(85, 15)
            failure_rate = np.random.beta(5, 95)
            bugs_detected = np.random.beta(10, 90)
            
            total_tests = np.random.poisson(500)
            failed_tests = int(total_tests * failure_rate)
            
            metrics = TestingMetrics(
                test_coverage=coverage,
                failure_rate=failure_rate,
                bugs_detected_percentage=bugs_detected,
                total_tests=total_tests,
                passed_tests=total_tests - failed_tests,
                failed_tests=failed_tests,
                test_duration=np.random.exponential(300)
            )
            
            self.metrics_history["testing"].append(metrics)
            return metrics
            
        except Exception as e:
            logger.error(f"Error collecting testing metrics: {e}")
            return TestingMetrics()
    
    def get_metrics_summary(self, hours: int = 24) -> Dict[str, Any]:
        """Get metrics summary for the last N hours"""
        
        cutoff_time = datetime.now() - timedelta(hours=hours)
        
        summary = {}
        
        for metric_type, history in self.metrics_history.items():
            if not history:
                continue
            
            # Filter by time
            recent_metrics = [
                m for m in history 
                if m.timestamp >= cutoff_time
            ]
            
            if not recent_metrics:
                continue
            
            # Calculate summary statistics
            if metric_type == "backend_api":
                summary[metric_type] = {
                    "avg_response_time": np.mean([m.response_time for m in recent_metrics]),
                    "avg_error_rate": np.mean([m.error_rate for m in recent_metrics]),
                    "avg_cpu_utilization": np.mean([m.cpu_utilization for m in recent_metrics]),
                    "avg_memory_utilization": np.mean([m.memory_utilization for m in recent_metrics]),
                    "avg_requests_per_second": np.mean([m.requests_per_second for m in recent_metrics])
                }
            
            elif metric_type == "pipeline":
                summary[metric_type] = {
                    "avg_success_rate": np.mean([m.dag_execution_success_rate for m in recent_metrics]),
                    "avg_failure_rate": np.mean([m.task_failure_rate for m in recent_metrics]),
                    "avg_duration": np.mean([m.average_task_duration for m in recent_metrics]),
                    "total_tasks": sum([m.total_tasks_executed for m in recent_metrics])
                }
            
            elif metric_type == "code":
                summary[metric_type] = {
                    "avg_cyclomatic_complexity": np.mean([m.cyclomatic_complexity for m in recent_metrics]),
                    "avg_duplication_rate": np.mean([m.duplication_rate for m in recent_metrics]),
                    "avg_technical_debt": np.mean([m.technical_debt for m in recent_metrics]),
                    "avg_test_coverage": np.mean([m.test_coverage for m in recent_metrics])
                }
            
            elif metric_type == "data":
                summary[metric_type] = {
                    "avg_query_latency": np.mean([m.avg_query_latency for m in recent_metrics]),
                    "p95_query_latency": np.mean([m.p95_query_latency for m in recent_metrics]),
                    "p99_query_latency": np.mean([m.p99_query_latency for m in recent_metrics]),
                    "avg_disk_usage": np.mean([m.disk_usage for m in recent_metrics])
                }
            
            elif metric_type == "scraping":
                summary[metric_type] = {
                    "avg_success_rate": np.mean([m.success_rate for m in recent_metrics]),
                    "avg_throughput": np.mean([m.scraping_throughput for m in recent_metrics]),
                    "avg_response_time": np.mean([m.response_time for m in recent_metrics]),
                    "total_pages_scraped": sum([m.pages_scraped_successfully for m in recent_metrics])
                }
            
            elif metric_type == "ml":
                summary[metric_type] = {
                    "avg_inference_time": np.mean([m.inference_time for m in recent_metrics]),
                    "avg_rmse": np.mean([m.rmse for m in recent_metrics]),
                    "avg_auc": np.mean([m.auc for m in recent_metrics]),
                    "avg_f1_score": np.mean([m.f1_score for m in recent_metrics])
                }
            
            elif metric_type == "testing":
                summary[metric_type] = {
                    "avg_coverage": np.mean([m.test_coverage for m in recent_metrics]),
                    "avg_failure_rate": np.mean([m.failure_rate for m in recent_metrics]),
                    "total_tests": sum([m.total_tests for m in recent_metrics]),
                    "total_passed": sum([m.passed_tests for m in recent_metrics])
                }
        
        return summary
    
    def generate_weekly_report(self) -> Dict[str, Any]:
        """Generate comprehensive weekly report"""
        
        report = {
            "report_date": datetime.now().isoformat(),
            "period": "weekly",
            "metrics": self.get_metrics_summary(hours=168),  # 7 days
            "alerts": self._generate_alerts(),
            "recommendations": self._generate_recommendations()
        }
        
        return report
    
    def _generate_alerts(self) -> List[Dict[str, Any]]:
        """Generate alerts based on metrics thresholds"""
        
        alerts = []
        summary = self.get_metrics_summary(hours=24)
        
        # Backend API alerts
        if "backend_api" in summary:
            api_metrics = summary["backend_api"]
            if api_metrics["avg_response_time"] > 1.0:
                alerts.append({
                    "type": "warning",
                    "category": "backend_api",
                    "message": f"High response time: {api_metrics['avg_response_time']:.2f}s",
                    "severity": "medium"
                })
            
            if api_metrics["avg_error_rate"] > 0.05:
                alerts.append({
                    "type": "error",
                    "category": "backend_api",
                    "message": f"High error rate: {api_metrics['avg_error_rate']:.2%}",
                    "severity": "high"
                })
        
        # Pipeline alerts
        if "pipeline" in summary:
            pipeline_metrics = summary["pipeline"]
            if pipeline_metrics["avg_failure_rate"] > 0.1:
                alerts.append({
                    "type": "error",
                    "category": "pipeline",
                    "message": f"High task failure rate: {pipeline_metrics['avg_failure_rate']:.2%}",
                    "severity": "high"
                })
        
        # Code quality alerts
        if "code" in summary:
            code_metrics = summary["code"]
            if code_metrics["avg_cyclomatic_complexity"] > 10:
                alerts.append({
                    "type": "warning",
                    "category": "code",
                    "message": f"High cyclomatic complexity: {code_metrics['avg_cyclomatic_complexity']:.2f}",
                    "severity": "medium"
                })
            
            if code_metrics["avg_test_coverage"] < 0.8:
                alerts.append({
                    "type": "warning",
                    "category": "code",
                    "message": f"Low test coverage: {code_metrics['avg_test_coverage']:.2%}",
                    "severity": "medium"
                })
        
        return alerts
    
    def _generate_recommendations(self) -> List[Dict[str, Any]]:
        """Generate recommendations based on metrics analysis"""
        
        recommendations = []
        summary = self.get_metrics_summary(hours=24)
        
        # Performance recommendations
        if "backend_api" in summary:
            api_metrics = summary["backend_api"]
            if api_metrics["avg_cpu_utilization"] > 80:
                recommendations.append({
                    "category": "performance",
                    "priority": "high",
                    "recommendation": "Consider scaling up CPU resources or optimizing code"
                })
            
            if api_metrics["avg_memory_utilization"] > 85:
                recommendations.append({
                    "category": "performance",
                    "priority": "high",
                    "recommendation": "Consider increasing memory allocation or optimizing memory usage"
                })
        
        # Code quality recommendations
        if "code" in summary:
            code_metrics = summary["code"]
            if code_metrics["avg_duplication_rate"] > 0.05:
                recommendations.append({
                    "category": "code_quality",
                    "priority": "medium",
                    "recommendation": "Refactor duplicated code to improve maintainability"
                })
            
            if code_metrics["avg_technical_debt"] > 50:
                recommendations.append({
                    "category": "code_quality",
                    "priority": "medium",
                    "recommendation": "Address technical debt to improve code quality"
                })
        
        return recommendations
    
    def save_metrics(self, file_path: str):
        """Save metrics to file"""
        
        metrics_data = {
            "timestamp": datetime.now().isoformat(),
            "metrics": {}
        }
        
        for metric_type, history in self.metrics_history.items():
            if history:
                metrics_data["metrics"][metric_type] = [
                    {
                        "timestamp": m.timestamp.isoformat(),
                        "data": m.__dict__
                    }
                    for m in history
                ]
        
        with open(file_path, "w") as f:
            json.dump(metrics_data, f, indent=2, default=str)
        
        logger.info(f"Metrics saved to {file_path}")
    
    def load_metrics(self, file_path: str):
        """Load metrics from file"""
        
        if not os.path.exists(file_path):
            logger.warning(f"Metrics file not found: {file_path}")
            return
        
        with open(file_path, "r") as f:
            metrics_data = json.load(f)
        
        # Clear existing metrics
        for history in self.metrics_history.values():
            history.clear()
        
        # Load metrics
        for metric_type, metrics_list in metrics_data.get("metrics", {}).items():
            if metric_type in self.metrics_history:
                for metric_data in metrics_list:
                    timestamp = datetime.fromisoformat(metric_data["timestamp"])
                    data = metric_data["data"]
                    
                    # Create appropriate metrics object
                    if metric_type == "backend_api":
                        metric = BackendAPIMetrics(**data)
                    elif metric_type == "pipeline":
                        metric = PipelineMetrics(**data)
                    elif metric_type == "code":
                        metric = CodeMetrics(**data)
                    elif metric_type == "data":
                        metric = DataMetrics(**data)
                    elif metric_type == "scraping":
                        metric = ScrapingMetrics(**data)
                    elif metric_type == "ml":
                        metric = MLMetrics(**data)
                    elif metric_type == "testing":
                        metric = TestingMetrics(**data)
                    else:
                        continue
                    
                    self.metrics_history[metric_type].append(metric)
        
        logger.info(f"Metrics loaded from {file_path}")


def main():
    """Test the metrics collector"""
    
    # Create metrics collector
    collector = MetricsCollector()
    
    # Collect sample metrics
    print("Collecting sample metrics...")
    
    backend_metrics = collector.collect_backend_api_metrics()
    pipeline_metrics = collector.collect_pipeline_metrics()
    code_metrics = collector.collect_code_metrics()
    data_metrics = collector.collect_data_metrics()
    scraping_metrics = collector.collect_scraping_metrics()
    ml_metrics = collector.collect_ml_metrics()
    testing_metrics = collector.collect_testing_metrics()
    
    # Generate summary
    summary = collector.get_metrics_summary(hours=1)
    print(f"Metrics summary: {json.dumps(summary, indent=2, default=str)}")
    
    # Generate weekly report
    weekly_report = collector.generate_weekly_report()
    print(f"Weekly report generated with {len(weekly_report['alerts'])} alerts and {len(weekly_report['recommendations'])} recommendations")
    
    # Save metrics
    collector.save_metrics("metrics_test.json")
    print("Metrics saved to metrics_test.json")


if __name__ == "__main__":
    main()
