"""
Weekly Reporting System for WealthArena Trading System

This module provides comprehensive weekly reporting capabilities including:
- Automated report generation
- Metrics visualization
- Performance analysis
- Alert generation
- Recommendations
- Export to various formats (PDF, HTML, JSON)
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
import logging
from datetime import datetime, timedelta
import json
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from dataclasses import dataclass
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import weasyprint
from jinja2 import Template
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.base import MIMEBase
from email import encoders

from ..metrics.comprehensive_metrics import MetricsCollector, BackendAPIMetrics, PipelineMetrics, CodeMetrics, DataMetrics, ScrapingMetrics, MLMetrics, TestingMetrics

logger = logging.getLogger(__name__)


@dataclass
class ReportConfig:
    """Configuration for weekly reports"""
    report_title: str = "WealthArena Weekly Report"
    company_name: str = "WealthArena Trading System"
    report_recipients: List[str] = None
    email_config: Dict[str, str] = None
    output_formats: List[str] = None
    include_charts: bool = True
    include_recommendations: bool = True
    include_alerts: bool = True


class WeeklyReportGenerator:
    """Weekly report generator for WealthArena system"""
    
    def __init__(self, config: ReportConfig = None):
        self.config = config or ReportConfig()
        self.metrics_collector = MetricsCollector()
        self.report_data = {}
        self.charts = {}
        
        # Set up output directory
        self.output_dir = Path("reports/weekly")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info("Weekly report generator initialized")
    
    def generate_report(self, week_start: datetime = None) -> Dict[str, Any]:
        """Generate comprehensive weekly report"""
        
        if week_start is None:
            week_start = datetime.now() - timedelta(days=7)
        
        logger.info(f"Generating weekly report for week starting {week_start.date()}")
        
        # Collect metrics
        self._collect_metrics()
        
        # Generate report sections
        report = {
            "metadata": self._generate_metadata(week_start),
            "executive_summary": self._generate_executive_summary(),
            "backend_api_metrics": self._generate_backend_api_section(),
            "pipeline_metrics": self._generate_pipeline_section(),
            "code_metrics": self._generate_code_section(),
            "data_metrics": self._generate_data_section(),
            "scraping_metrics": self._generate_scraping_section(),
            "ml_metrics": self._generate_ml_section(),
            "testing_metrics": self._generate_testing_section(),
            "alerts": self._generate_alerts_section(),
            "recommendations": self._generate_recommendations_section(),
            "charts": self._generate_charts() if self.config.include_charts else {},
            "appendix": self._generate_appendix()
        }
        
        self.report_data = report
        
        # Save report
        self._save_report(report)
        
        logger.info("Weekly report generated successfully")
        return report
    
    def _collect_metrics(self):
        """Collect all metrics for the report"""
        
        # Collect current metrics
        self.metrics_collector.collect_backend_api_metrics()
        self.metrics_collector.collect_pipeline_metrics()
        self.metrics_collector.collect_code_metrics()
        self.metrics_collector.collect_data_metrics()
        self.metrics_collector.collect_scraping_metrics()
        self.metrics_collector.collect_ml_metrics()
        self.metrics_collector.collect_testing_metrics()
    
    def _generate_metadata(self, week_start: datetime) -> Dict[str, Any]:
        """Generate report metadata"""
        
        week_end = week_start + timedelta(days=6)
        
        return {
            "report_title": self.config.report_title,
            "company_name": self.config.company_name,
            "report_date": datetime.now().isoformat(),
            "report_period": {
                "start": week_start.isoformat(),
                "end": week_end.isoformat()
            },
            "generated_by": "WealthArena Reporting System",
            "version": "1.0.0"
        }
    
    def _generate_executive_summary(self) -> Dict[str, Any]:
        """Generate executive summary"""
        
        summary = self.metrics_collector.get_metrics_summary(hours=168)  # 7 days
        
        # Calculate overall health score
        health_score = self._calculate_health_score(summary)
        
        # Generate key insights
        insights = self._generate_key_insights(summary)
        
        return {
            "health_score": health_score,
            "key_insights": insights,
            "overall_status": "healthy" if health_score > 80 else "warning" if health_score > 60 else "critical",
            "summary": f"System health score: {health_score:.1f}/100. {len(insights)} key insights identified."
        }
    
    def _calculate_health_score(self, summary: Dict[str, Any]) -> float:
        """Calculate overall system health score"""
        
        score = 100.0
        
        # Backend API health
        if "backend_api" in summary:
            api_metrics = summary["backend_api"]
            if api_metrics["avg_response_time"] > 1.0:
                score -= 10
            if api_metrics["avg_error_rate"] > 0.05:
                score -= 20
            if api_metrics["avg_cpu_utilization"] > 80:
                score -= 10
            if api_metrics["avg_memory_utilization"] > 85:
                score -= 10
        
        # Pipeline health
        if "pipeline" in summary:
            pipeline_metrics = summary["pipeline"]
            if pipeline_metrics["avg_failure_rate"] > 0.1:
                score -= 15
            if pipeline_metrics["avg_duration"] > 60:
                score -= 5
        
        # Code quality health
        if "code" in summary:
            code_metrics = summary["code"]
            if code_metrics["avg_test_coverage"] < 0.8:
                score -= 10
            if code_metrics["avg_cyclomatic_complexity"] > 10:
                score -= 5
            if code_metrics["avg_duplication_rate"] > 0.05:
                score -= 5
        
        # Data health
        if "data" in summary:
            data_metrics = summary["data"]
            if data_metrics["avg_query_latency"] > 0.1:
                score -= 5
            if data_metrics["avg_disk_usage"] > 90:
                score -= 10
        
        return max(0, score)
    
    def _generate_key_insights(self, summary: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate key insights from metrics"""
        
        insights = []
        
        # Backend API insights
        if "backend_api" in summary:
            api_metrics = summary["backend_api"]
            if api_metrics["avg_response_time"] > 0.5:
                insights.append({
                    "category": "Performance",
                    "insight": f"API response time is {api_metrics['avg_response_time']:.2f}s, above optimal threshold",
                    "impact": "medium",
                    "recommendation": "Consider optimizing API endpoints or scaling resources"
                })
            
            if api_metrics["avg_error_rate"] > 0.01:
                insights.append({
                    "category": "Reliability",
                    "insight": f"Error rate is {api_metrics['avg_error_rate']:.2%}, above acceptable threshold",
                    "impact": "high",
                    "recommendation": "Investigate and fix error sources"
                })
        
        # Pipeline insights
        if "pipeline" in summary:
            pipeline_metrics = summary["pipeline"]
            if pipeline_metrics["avg_failure_rate"] > 0.05:
                insights.append({
                    "category": "Pipeline",
                    "insight": f"Pipeline failure rate is {pipeline_metrics['avg_failure_rate']:.2%}",
                    "impact": "high",
                    "recommendation": "Review and fix failing pipeline tasks"
                })
        
        # Code quality insights
        if "code" in summary:
            code_metrics = summary["code"]
            if code_metrics["avg_test_coverage"] < 0.8:
                insights.append({
                    "category": "Code Quality",
                    "insight": f"Test coverage is {code_metrics['avg_test_coverage']:.1%}, below recommended 80%",
                    "impact": "medium",
                    "recommendation": "Increase test coverage to improve code reliability"
                })
        
        return insights
    
    def _generate_backend_api_section(self) -> Dict[str, Any]:
        """Generate backend API metrics section"""
        
        summary = self.metrics_collector.get_metrics_summary(hours=168)
        api_metrics = summary.get("backend_api", {})
        
        return {
            "title": "Backend API Performance",
            "metrics": {
                "response_time": {
                    "value": api_metrics.get("avg_response_time", 0),
                    "unit": "seconds",
                    "status": "good" if api_metrics.get("avg_response_time", 0) < 0.5 else "warning" if api_metrics.get("avg_response_time", 0) < 1.0 else "critical"
                },
                "error_rate": {
                    "value": api_metrics.get("avg_error_rate", 0),
                    "unit": "percentage",
                    "status": "good" if api_metrics.get("avg_error_rate", 0) < 0.01 else "warning" if api_metrics.get("avg_error_rate", 0) < 0.05 else "critical"
                },
                "cpu_utilization": {
                    "value": api_metrics.get("avg_cpu_utilization", 0),
                    "unit": "percentage",
                    "status": "good" if api_metrics.get("avg_cpu_utilization", 0) < 70 else "warning" if api_metrics.get("avg_cpu_utilization", 0) < 80 else "critical"
                },
                "memory_utilization": {
                    "value": api_metrics.get("avg_memory_utilization", 0),
                    "unit": "percentage",
                    "status": "good" if api_metrics.get("avg_memory_utilization", 0) < 75 else "warning" if api_metrics.get("avg_memory_utilization", 0) < 85 else "critical"
                },
                "requests_per_second": {
                    "value": api_metrics.get("avg_requests_per_second", 0),
                    "unit": "requests/second",
                    "status": "good"
                }
            },
            "trends": self._calculate_trends("backend_api"),
            "recommendations": self._get_backend_recommendations(api_metrics)
        }
    
    def _generate_pipeline_section(self) -> Dict[str, Any]:
        """Generate pipeline metrics section"""
        
        summary = self.metrics_collector.get_metrics_summary(hours=168)
        pipeline_metrics = summary.get("pipeline", {})
        
        return {
            "title": "Pipeline Performance",
            "metrics": {
                "success_rate": {
                    "value": pipeline_metrics.get("avg_success_rate", 0),
                    "unit": "percentage",
                    "status": "good" if pipeline_metrics.get("avg_success_rate", 0) > 0.95 else "warning" if pipeline_metrics.get("avg_success_rate", 0) > 0.9 else "critical"
                },
                "failure_rate": {
                    "value": pipeline_metrics.get("avg_failure_rate", 0),
                    "unit": "percentage",
                    "status": "good" if pipeline_metrics.get("avg_failure_rate", 0) < 0.05 else "warning" if pipeline_metrics.get("avg_failure_rate", 0) < 0.1 else "critical"
                },
                "average_duration": {
                    "value": pipeline_metrics.get("avg_duration", 0),
                    "unit": "seconds",
                    "status": "good" if pipeline_metrics.get("avg_duration", 0) < 30 else "warning" if pipeline_metrics.get("avg_duration", 0) < 60 else "critical"
                },
                "total_tasks": {
                    "value": pipeline_metrics.get("total_tasks", 0),
                    "unit": "tasks",
                    "status": "good"
                }
            },
            "trends": self._calculate_trends("pipeline"),
            "recommendations": self._get_pipeline_recommendations(pipeline_metrics)
        }
    
    def _generate_code_section(self) -> Dict[str, Any]:
        """Generate code quality metrics section"""
        
        summary = self.metrics_collector.get_metrics_summary(hours=168)
        code_metrics = summary.get("code", {})
        
        return {
            "title": "Code Quality Metrics",
            "metrics": {
                "cyclomatic_complexity": {
                    "value": code_metrics.get("avg_cyclomatic_complexity", 0),
                    "unit": "complexity",
                    "status": "good" if code_metrics.get("avg_cyclomatic_complexity", 0) < 5 else "warning" if code_metrics.get("avg_cyclomatic_complexity", 0) < 10 else "critical"
                },
                "duplication_rate": {
                    "value": code_metrics.get("avg_duplication_rate", 0),
                    "unit": "percentage",
                    "status": "good" if code_metrics.get("avg_duplication_rate", 0) < 0.03 else "warning" if code_metrics.get("avg_duplication_rate", 0) < 0.05 else "critical"
                },
                "technical_debt": {
                    "value": code_metrics.get("avg_technical_debt", 0),
                    "unit": "hours",
                    "status": "good" if code_metrics.get("avg_technical_debt", 0) < 20 else "warning" if code_metrics.get("avg_technical_debt", 0) < 50 else "critical"
                },
                "test_coverage": {
                    "value": code_metrics.get("avg_test_coverage", 0),
                    "unit": "percentage",
                    "status": "good" if code_metrics.get("avg_test_coverage", 0) > 0.8 else "warning" if code_metrics.get("avg_test_coverage", 0) > 0.6 else "critical"
                }
            },
            "trends": self._calculate_trends("code"),
            "recommendations": self._get_code_recommendations(code_metrics)
        }
    
    def _generate_data_section(self) -> Dict[str, Any]:
        """Generate data store metrics section"""
        
        summary = self.metrics_collector.get_metrics_summary(hours=168)
        data_metrics = summary.get("data", {})
        
        return {
            "title": "Data Store Performance",
            "metrics": {
                "avg_query_latency": {
                    "value": data_metrics.get("avg_query_latency", 0),
                    "unit": "seconds",
                    "status": "good" if data_metrics.get("avg_query_latency", 0) < 0.05 else "warning" if data_metrics.get("avg_query_latency", 0) < 0.1 else "critical"
                },
                "p95_query_latency": {
                    "value": data_metrics.get("p95_query_latency", 0),
                    "unit": "seconds",
                    "status": "good" if data_metrics.get("p95_query_latency", 0) < 0.1 else "warning" if data_metrics.get("p95_query_latency", 0) < 0.2 else "critical"
                },
                "p99_query_latency": {
                    "value": data_metrics.get("p99_query_latency", 0),
                    "unit": "seconds",
                    "status": "good" if data_metrics.get("p99_query_latency", 0) < 0.2 else "warning" if data_metrics.get("p99_query_latency", 0) < 0.5 else "critical"
                },
                "disk_usage": {
                    "value": data_metrics.get("avg_disk_usage", 0),
                    "unit": "percentage",
                    "status": "good" if data_metrics.get("avg_disk_usage", 0) < 70 else "warning" if data_metrics.get("avg_disk_usage", 0) < 90 else "critical"
                }
            },
            "trends": self._calculate_trends("data"),
            "recommendations": self._get_data_recommendations(data_metrics)
        }
    
    def _generate_scraping_section(self) -> Dict[str, Any]:
        """Generate scraping metrics section"""
        
        summary = self.metrics_collector.get_metrics_summary(hours=168)
        scraping_metrics = summary.get("scraping", {})
        
        return {
            "title": "Web Scraping Performance",
            "metrics": {
                "success_rate": {
                    "value": scraping_metrics.get("avg_success_rate", 0),
                    "unit": "percentage",
                    "status": "good" if scraping_metrics.get("avg_success_rate", 0) > 0.9 else "warning" if scraping_metrics.get("avg_success_rate", 0) > 0.8 else "critical"
                },
                "throughput": {
                    "value": scraping_metrics.get("avg_throughput", 0),
                    "unit": "pages/hour",
                    "status": "good" if scraping_metrics.get("avg_throughput", 0) > 100 else "warning" if scraping_metrics.get("avg_throughput", 0) > 50 else "critical"
                },
                "response_time": {
                    "value": scraping_metrics.get("avg_response_time", 0),
                    "unit": "seconds",
                    "status": "good" if scraping_metrics.get("avg_response_time", 0) < 1.0 else "warning" if scraping_metrics.get("avg_response_time", 0) < 2.0 else "critical"
                },
                "pages_scraped": {
                    "value": scraping_metrics.get("total_pages_scraped", 0),
                    "unit": "pages",
                    "status": "good"
                }
            },
            "trends": self._calculate_trends("scraping"),
            "recommendations": self._get_scraping_recommendations(scraping_metrics)
        }
    
    def _generate_ml_section(self) -> Dict[str, Any]:
        """Generate ML metrics section"""
        
        summary = self.metrics_collector.get_metrics_summary(hours=168)
        ml_metrics = summary.get("ml", {})
        
        return {
            "title": "Machine Learning Performance",
            "metrics": {
                "inference_time": {
                    "value": ml_metrics.get("avg_inference_time", 0),
                    "unit": "seconds",
                    "status": "good" if ml_metrics.get("avg_inference_time", 0) < 0.01 else "warning" if ml_metrics.get("avg_inference_time", 0) < 0.05 else "critical"
                },
                "rmse": {
                    "value": ml_metrics.get("avg_rmse", 0),
                    "unit": "error",
                    "status": "good" if ml_metrics.get("avg_rmse", 0) < 0.1 else "warning" if ml_metrics.get("avg_rmse", 0) < 0.2 else "critical"
                },
                "auc": {
                    "value": ml_metrics.get("avg_auc", 0),
                    "unit": "score",
                    "status": "good" if ml_metrics.get("avg_auc", 0) > 0.8 else "warning" if ml_metrics.get("avg_auc", 0) > 0.7 else "critical"
                },
                "f1_score": {
                    "value": ml_metrics.get("avg_f1_score", 0),
                    "unit": "score",
                    "status": "good" if ml_metrics.get("avg_f1_score", 0) > 0.8 else "warning" if ml_metrics.get("avg_f1_score", 0) > 0.7 else "critical"
                }
            },
            "trends": self._calculate_trends("ml"),
            "recommendations": self._get_ml_recommendations(ml_metrics)
        }
    
    def _generate_testing_section(self) -> Dict[str, Any]:
        """Generate testing metrics section"""
        
        summary = self.metrics_collector.get_metrics_summary(hours=168)
        testing_metrics = summary.get("testing", {})
        
        return {
            "title": "Testing Performance",
            "metrics": {
                "coverage": {
                    "value": testing_metrics.get("avg_coverage", 0),
                    "unit": "percentage",
                    "status": "good" if testing_metrics.get("avg_coverage", 0) > 0.8 else "warning" if testing_metrics.get("avg_coverage", 0) > 0.6 else "critical"
                },
                "failure_rate": {
                    "value": testing_metrics.get("avg_failure_rate", 0),
                    "unit": "percentage",
                    "status": "good" if testing_metrics.get("avg_failure_rate", 0) < 0.05 else "warning" if testing_metrics.get("avg_failure_rate", 0) < 0.1 else "critical"
                },
                "total_tests": {
                    "value": testing_metrics.get("total_tests", 0),
                    "unit": "tests",
                    "status": "good"
                },
                "passed_tests": {
                    "value": testing_metrics.get("total_passed", 0),
                    "unit": "tests",
                    "status": "good"
                }
            },
            "trends": self._calculate_trends("testing"),
            "recommendations": self._get_testing_recommendations(testing_metrics)
        }
    
    def _generate_alerts_section(self) -> Dict[str, Any]:
        """Generate alerts section"""
        
        alerts = self.metrics_collector._generate_alerts()
        
        return {
            "title": "System Alerts",
            "total_alerts": len(alerts),
            "alerts_by_severity": {
                "critical": len([a for a in alerts if a.get("severity") == "high"]),
                "warning": len([a for a in alerts if a.get("severity") == "medium"]),
                "info": len([a for a in alerts if a.get("severity") == "low"])
            },
            "alerts": alerts
        }
    
    def _generate_recommendations_section(self) -> Dict[str, Any]:
        """Generate recommendations section"""
        
        recommendations = self.metrics_collector._generate_recommendations()
        
        return {
            "title": "Recommendations",
            "total_recommendations": len(recommendations),
            "recommendations_by_priority": {
                "high": len([r for r in recommendations if r.get("priority") == "high"]),
                "medium": len([r for r in recommendations if r.get("priority") == "medium"]),
                "low": len([r for r in recommendations if r.get("priority") == "low"])
            },
            "recommendations": recommendations
        }
    
    def _generate_charts(self) -> Dict[str, Any]:
        """Generate charts for the report"""
        
        charts = {}
        
        # Backend API response time chart
        charts["backend_api_response_time"] = self._create_response_time_chart()
        
        # Pipeline success rate chart
        charts["pipeline_success_rate"] = self._create_pipeline_success_chart()
        
        # Code quality trends chart
        charts["code_quality_trends"] = self._create_code_quality_chart()
        
        # ML performance chart
        charts["ml_performance"] = self._create_ml_performance_chart()
        
        return charts
    
    def _create_response_time_chart(self) -> str:
        """Create response time chart"""
        
        # This would create an actual chart in a real implementation
        return "backend_api_response_time_chart.png"
    
    def _create_pipeline_success_chart(self) -> str:
        """Create pipeline success rate chart"""
        
        return "pipeline_success_rate_chart.png"
    
    def _create_code_quality_chart(self) -> str:
        """Create code quality trends chart"""
        
        return "code_quality_trends_chart.png"
    
    def _create_ml_performance_chart(self) -> str:
        """Create ML performance chart"""
        
        return "ml_performance_chart.png"
    
    def _calculate_trends(self, metric_type: str) -> Dict[str, Any]:
        """Calculate trends for a metric type"""
        
        # This would calculate actual trends in a real implementation
        return {
            "direction": "stable",
            "change_percentage": 0.0,
            "trend_description": "No significant change"
        }
    
    def _get_backend_recommendations(self, metrics: Dict[str, Any]) -> List[str]:
        """Get backend API recommendations"""
        
        recommendations = []
        
        if metrics.get("avg_response_time", 0) > 0.5:
            recommendations.append("Consider optimizing API endpoints or implementing caching")
        
        if metrics.get("avg_error_rate", 0) > 0.01:
            recommendations.append("Investigate and fix error sources")
        
        if metrics.get("avg_cpu_utilization", 0) > 80:
            recommendations.append("Consider scaling up CPU resources")
        
        return recommendations
    
    def _get_pipeline_recommendations(self, metrics: Dict[str, Any]) -> List[str]:
        """Get pipeline recommendations"""
        
        recommendations = []
        
        if metrics.get("avg_failure_rate", 0) > 0.05:
            recommendations.append("Review and fix failing pipeline tasks")
        
        if metrics.get("avg_duration", 0) > 60:
            recommendations.append("Optimize pipeline task performance")
        
        return recommendations
    
    def _get_code_recommendations(self, metrics: Dict[str, Any]) -> List[str]:
        """Get code quality recommendations"""
        
        recommendations = []
        
        if metrics.get("avg_test_coverage", 0) < 0.8:
            recommendations.append("Increase test coverage to improve code reliability")
        
        if metrics.get("avg_duplication_rate", 0) > 0.05:
            recommendations.append("Refactor duplicated code to improve maintainability")
        
        if metrics.get("avg_cyclomatic_complexity", 0) > 10:
            recommendations.append("Reduce cyclomatic complexity by breaking down complex functions")
        
        return recommendations
    
    def _get_data_recommendations(self, metrics: Dict[str, Any]) -> List[str]:
        """Get data store recommendations"""
        
        recommendations = []
        
        if metrics.get("avg_query_latency", 0) > 0.05:
            recommendations.append("Optimize database queries and consider indexing")
        
        if metrics.get("avg_disk_usage", 0) > 90:
            recommendations.append("Consider data archiving or disk expansion")
        
        return recommendations
    
    def _get_scraping_recommendations(self, metrics: Dict[str, Any]) -> List[str]:
        """Get scraping recommendations"""
        
        recommendations = []
        
        if metrics.get("avg_success_rate", 0) < 0.9:
            recommendations.append("Improve scraping success rate by handling errors better")
        
        if metrics.get("avg_throughput", 0) < 100:
            recommendations.append("Optimize scraping throughput by improving concurrency")
        
        return recommendations
    
    def _get_ml_recommendations(self, metrics: Dict[str, Any]) -> List[str]:
        """Get ML recommendations"""
        
        recommendations = []
        
        if metrics.get("avg_inference_time", 0) > 0.01:
            recommendations.append("Optimize model inference time")
        
        if metrics.get("avg_rmse", 0) > 0.1:
            recommendations.append("Improve model accuracy by retraining with more data")
        
        return recommendations
    
    def _get_testing_recommendations(self, metrics: Dict[str, Any]) -> List[str]:
        """Get testing recommendations"""
        
        recommendations = []
        
        if metrics.get("avg_coverage", 0) < 0.8:
            recommendations.append("Increase test coverage")
        
        if metrics.get("avg_failure_rate", 0) > 0.05:
            recommendations.append("Investigate and fix failing tests")
        
        return recommendations
    
    def _generate_appendix(self) -> Dict[str, Any]:
        """Generate report appendix"""
        
        return {
            "title": "Appendix",
            "sections": {
                "methodology": "This report uses automated metrics collection and analysis",
                "data_sources": "System metrics, logs, and performance data",
                "calculation_methods": "Statistical analysis and trend calculation",
                "definitions": {
                    "health_score": "Overall system health score (0-100)",
                    "response_time": "Average API response time in seconds",
                    "error_rate": "Percentage of failed requests",
                    "test_coverage": "Percentage of code covered by tests"
                }
            }
        }
    
    def _save_report(self, report: Dict[str, Any]):
        """Save report to files"""
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save JSON report
        json_path = self.output_dir / f"weekly_report_{timestamp}.json"
        with open(json_path, "w") as f:
            json.dump(report, f, indent=2, default=str)
        
        # Save HTML report
        html_path = self.output_dir / f"weekly_report_{timestamp}.html"
        self._generate_html_report(report, html_path)
        
        # Save PDF report
        pdf_path = self.output_dir / f"weekly_report_{timestamp}.pdf"
        self._generate_pdf_report(report, pdf_path)
        
        logger.info(f"Report saved to {self.output_dir}")
    
    def _generate_html_report(self, report: Dict[str, Any], output_path: Path):
        """Generate HTML report"""
        
        html_template = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>{{ report.metadata.report_title }}</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 40px; }
                .header { text-align: center; margin-bottom: 40px; }
                .section { margin-bottom: 30px; }
                .metric { display: inline-block; margin: 10px; padding: 10px; border: 1px solid #ccc; }
                .status-good { background-color: #d4edda; }
                .status-warning { background-color: #fff3cd; }
                .status-critical { background-color: #f8d7da; }
            </style>
        </head>
        <body>
            <div class="header">
                <h1>{{ report.metadata.report_title }}</h1>
                <p>{{ report.metadata.company_name }}</p>
                <p>Report Period: {{ report.metadata.report_period.start }} to {{ report.metadata.report_period.end }}</p>
            </div>
            
            <div class="section">
                <h2>Executive Summary</h2>
                <p>{{ report.executive_summary.summary }}</p>
                <p>Health Score: {{ report.executive_summary.health_score }}</p>
            </div>
            
            <!-- Add more sections as needed -->
        </body>
        </html>
        """
        
        template = Template(html_template)
        html_content = template.render(report=report)
        
        with open(output_path, "w") as f:
            f.write(html_content)
    
    def _generate_pdf_report(self, report: Dict[str, Any], output_path: Path):
        """Generate PDF report"""
        
        # This would generate a PDF in a real implementation
        # For now, we'll just create a placeholder
        with open(output_path, "w") as f:
            f.write("PDF report placeholder")
    
    def send_email_report(self, report: Dict[str, Any]):
        """Send report via email"""
        
        if not self.config.email_config or not self.config.report_recipients:
            logger.warning("Email configuration not provided")
            return
        
        try:
            # Create email message
            msg = MIMEMultipart()
            msg['From'] = self.config.email_config['from_email']
            msg['To'] = ', '.join(self.config.report_recipients)
            msg['Subject'] = f"Weekly Report - {datetime.now().strftime('%Y-%m-%d')}"
            
            # Add body
            body = f"""
            Weekly Report Summary:
            
            Health Score: {report['executive_summary']['health_score']}
            Status: {report['executive_summary']['overall_status']}
            
            Key Insights:
            {chr(10).join([f"- {insight['insight']}" for insight in report['executive_summary']['key_insights']])}
            
            Please see attached detailed report.
            """
            
            msg.attach(MIMEText(body, 'plain'))
            
            # Add attachment
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            json_path = self.output_dir / f"weekly_report_{timestamp}.json"
            
            if json_path.exists():
                with open(json_path, "rb") as f:
                    part = MIMEBase('application', 'octet-stream')
                    part.set_payload(f.read())
                    encoders.encode_base64(part)
                    part.add_header('Content-Disposition', f'attachment; filename=weekly_report_{timestamp}.json')
                    msg.attach(part)
            
            # Send email
            server = smtplib.SMTP(self.config.email_config['smtp_server'], self.config.email_config['smtp_port'])
            server.starttls()
            server.login(self.config.email_config['username'], self.config.email_config['password'])
            server.send_message(msg)
            server.quit()
            
            logger.info("Email report sent successfully")
            
        except Exception as e:
            logger.error(f"Error sending email report: {e}")


def main():
    """Test the weekly report generator"""
    
    # Create report configuration
    config = ReportConfig(
        report_title="WealthArena Weekly Report",
        company_name="WealthArena Trading System",
        report_recipients=["admin@wealtharena.com"],
        email_config={
            "smtp_server": "smtp.gmail.com",
            "smtp_port": 587,
            "username": "your_email@gmail.com",
            "password": "your_password",
            "from_email": "your_email@gmail.com"
        },
        output_formats=["json", "html", "pdf"],
        include_charts=True,
        include_recommendations=True,
        include_alerts=True
    )
    
    # Create report generator
    generator = WeeklyReportGenerator(config)
    
    # Generate report
    report = generator.generate_report()
    
    print("Weekly Report Generated Successfully!")
    print(f"Health Score: {report['executive_summary']['health_score']}")
    print(f"Status: {report['executive_summary']['overall_status']}")
    print(f"Key Insights: {len(report['executive_summary']['key_insights'])}")
    print(f"Alerts: {report['alerts']['total_alerts']}")
    print(f"Recommendations: {report['recommendations']['total_recommendations']}")


if __name__ == "__main__":
    main()
