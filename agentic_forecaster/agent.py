import pandas as pd
import logging
from typing import Optional, Any
from sktime.forecasting.base import BaseForecaster
from sktime_mcp.registry.interface import get_registry

logger = logging.getLogger(__name__)

class AgenticForecaster(BaseForecaster):
    """
    An agentic forecaster that selects and configures an sktime estimator
    based on a natural language prompt and data characteristics.
    """
    
    _tags = {
        "scitype:y": "both",
        "capability:pred_int": True,
        "requires-fh-in-fit": False,
        "X-y-must-have same-index": True,
    }

    def __init__(self, prompt: str, llm_client: Any = None):
        self.prompt = prompt
        self.llm_client = llm_client
        self.estimator_ = None
        self.selected_model_name_ = None
        self.explanation_ = None
        super().__init__()

    def _fit(self, y, X=None, fh=None):
        registry = get_registry()
        lower_prompt = self.prompt.lower()
        tags_to_query = {}
        
        if "interval" in lower_prompt or "probabilistic" in lower_prompt:
            tags_to_query["capability:pred_int"] = True
            
        if "multivariate" in lower_prompt:
            tags_to_query["scitype:y"] = "multivariate"
        else:
            tags_to_query["scitype:y"] = "univariate"
            
        estimators = registry.get_all_estimators(task="forecasting", tags=tags_to_query)
        if not estimators:
            estimators = registry.get_all_estimators(task="forecasting")
            
        if "arima" in lower_prompt:
            selected_node = next((e for e in estimators if "ARIMA" in e.name), estimators[0])
        else:
            selected_node = next((e for e in estimators if "AutoARIMA" in e.name), estimators[0])
            
        self.selected_model_name_ = selected_node.name
        self.estimator_ = selected_node.class_ref()
        self.explanation_ = f"Selected {self.selected_model_name_} for prompt: {self.prompt}"
        
        self.estimator_.fit(y, X=X, fh=fh)
        return self

    def _predict(self, fh=None, X=None):
        return self.estimator_.predict(fh=fh, X=X)

    def explain(self):
        return self.explanation_
