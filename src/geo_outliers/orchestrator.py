from __future__ import annotations

from dataclasses import dataclass, asdict
from datetime import datetime, timezone


@dataclass
class OrchestratorDecision:
    stage:str
    status:str
    action:str
    reason:str
    retryable:bool=False
    strategy:str|None=None


class MeridianOrchestrator:
    """Deterministic orchestration layer for Meridian Autopilot.

    It coordinates stages, quality gates and fallback decisions without hiding
    scientific assumptions. It does not invent data and never overrides hard
    validation failures.
    """

    def __init__(self, job_id:str):
        self.job_id=job_id
        self.started=datetime.now(timezone.utc).isoformat()
        self.decisions:list[OrchestratorDecision]=[]

    def record(self,stage,status,action,reason,retryable=False,strategy=None):
        d=OrchestratorDecision(stage,status,action,reason,retryable,strategy)
        self.decisions.append(d)
        return d

    def gate_source(self,gdf)->dict:
        if gdf is None or len(gdf)==0:
            self.record("source","blocked","stop","No observations were loaded")
            return {"ok":False}
        if getattr(gdf,"crs",None) is None:
            self.record("source","blocked","stop","Dataset has no CRS")
            return {"ok":False}
        if len(gdf)<30:
            self.record("source","blocked","stop","At least 30 observations are required")
            return {"ok":False}
        self.record("source","passed","continue",f"{len(gdf)} observations with CRS {gdf.crs}")
        return {"ok":True}

    def gate_autopilot(self,plan:dict)->dict:
        if not plan.get("ready"):
            reason="; ".join(plan.get("blockers") or ["Autopilot plan is not ready"])
            self.record("autopilot","blocked","stop",reason)
            return {"ok":False}
        conf=float((plan.get("domain_inference") or {}).get("confidence") or 0)
        if conf<.34 and plan.get("selected_domain")!="general":
            self.record("autopilot","warning","continue","Low-confidence domain inference; analysis remains explainable",False,"retain_selected_domain")
        else:
            self.record("autopilot","passed","continue",f"Domain {plan.get('selected_domain')} selected")
        return {"ok":True}

    def choose_analysis_strategy(self,rows:int,variables:list[str],quadrature_order:int)->dict:
        strategy={"quadrature_order":quadrature_order,"comparison_sample":10000,"bootstrap_n":3,"mode":"full"}
        if rows>150000:
            strategy.update({"quadrature_order":min(quadrature_order,32),"comparison_sample":6000,"bootstrap_n":2,"mode":"large_dataset"})
            self.record("strategy","adapted","continue","Large dataset profile selected to control runtime",False,"large_dataset")
        elif rows>60000:
            strategy.update({"comparison_sample":8000,"mode":"balanced"})
            self.record("strategy","adapted","continue","Balanced profile selected for medium-large dataset",False,"balanced")
        else:
            self.record("strategy","passed","continue","Full analysis profile selected",False,"full")
        if len(variables)>8:
            strategy["variables"]=variables[:8]
            self.record("strategy","adapted","continue","Variable set capped at 8 for stable multivariate geometry",False,"top_8_variables")
        return strategy

    def evaluate_distribution(self,summary:dict)->dict:
        m=summary.get("best_distribution_metrics") or {}
        p=m.get("ks_pvalue")
        if p is not None and p<.01:
            self.record("probability","warning","continue","Best-fit distribution still differs significantly by KS; ensemble evidence remains primary",False,"ensemble_consensus")
            return {"status":"warning","fallback":"ensemble_consensus"}
        self.record("probability","passed","continue","Probability fit accepted as one component of the ensemble")
        return {"status":"passed"}

    def evaluate_validation(self,summary:dict)->dict:
        val=summary.get("model_validation") or {}
        ablation=(val.get("ablation") or {})
        if isinstance(ablation,dict):
            values=[v for v in ablation.values() if isinstance(v,(int,float))]
            if values and max(values)>.35:
                self.record("validation","warning","continue","Sensitivity indicates one component may have high influence",False,"report_sensitivity")
                return {"status":"warning"}
        self.record("validation","passed","continue","Validation completed")
        return {"status":"passed"}

    def evaluate_enrichment(self,enrichment:dict)->dict:
        results=enrichment.get("results") or []
        errors=[r for r in results if r.get("status")=="error"]
        if errors:
            self.record("enrichment","degraded","continue",f"{len(errors)} external context provider(s) failed; core analysis preserved",True,"context_optional")
            return {"status":"degraded","core_analysis_preserved":True}
        self.record("enrichment","passed","continue","External context stage completed without blocking core analysis")
        return {"status":"passed"}

    def finalize(self)->dict:
        blocked=any(x.status=="blocked" for x in self.decisions)
        warnings=sum(x.status in {"warning","degraded","adapted"} for x in self.decisions)
        return {
            "job_id":self.job_id,
            "started_utc":self.started,
            "finished_utc":datetime.now(timezone.utc).isoformat(),
            "status":"blocked" if blocked else ("complete_with_warnings" if warnings else "complete"),
            "warnings":warnings,
            "decisions":[asdict(x) for x in self.decisions],
        }
