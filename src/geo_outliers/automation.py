from __future__ import annotations

from .domains import list_domain_packs, get_domain_pack, configure_domain
from .semantics import semantic_contract


def _norm(s:str)->str:
    return str(s).strip().lower().replace(" ","_").replace("-","_")


def infer_domain(columns:list[str]) -> dict:
    """Infer the most plausible Meridian Domain Pack from source field names.
    This is a configuration heuristic, not a scientific classification.
    """
    cols={_norm(c) for c in columns}
    ranked=[]
    for pack in list_domain_packs():
        if pack["id"]=="general":
            continue
        aliases=[]
        for group in pack.get("aliases") or []:
            aliases.extend(_norm(x) for x in group)
        aliases.extend(_norm(x) for x in pack.get("variables") or [])
        hits=sorted({a for a in aliases if a in cols})
        partial=sorted({a for a in aliases for c in cols if len(a)>=4 and (a in c or c in a)})
        score=len(hits)*2.0+len(partial)*0.6
        ranked.append({"domain":pack["id"],"name":pack["name"],"score":round(score,3),"exact_hits":hits,"partial_hits":partial[:12],"governance":pack["governance"]})
    ranked.sort(key=lambda x:x["score"],reverse=True)
    best=ranked[0] if ranked and ranked[0]["score"]>0 else {"domain":"general","name":"General Geospatial","score":0.0,"exact_hits":[],"partial_hits":[],"governance":"standard"}
    confidence=min(1.0,best["score"]/6.0)
    return {"recommended_domain":best["domain"],"confidence":round(confidence,3),"reason":"field-name semantic matching","ranking":ranked[:5]}


def build_autopilot_plan(df, requested_domain:str="auto") -> dict:
    columns=[str(c) for c in df.columns if str(c)!="geometry"]
    inference=infer_domain(columns)
    domain=inference["recommended_domain"] if requested_domain in ("","auto",None) else requested_domain
    pack=get_domain_pack(domain)
    contract=semantic_contract(df,pack)
    numeric=[p["column"] for p in contract["columns"] if p["usable_numeric"]]
    config=configure_domain(domain,numeric)
    blockers=[]
    warnings=list(contract.get("issues") or [])
    if not contract.get("ready"): blockers.append("Semantic contract is not ready for analysis")
    if config.get("probability_feature") is None: blockers.append("No usable numeric probability feature")
    if len(config.get("variables") or [])<2: blockers.append("Fewer than two usable numeric variables")
    if pack.get("governance")=="restricted":
        warnings.append("Restricted governance domain: operational use requires additional privacy/governance controls")
    return {
        "mode":"autopilot",
        "domain_inference":inference,
        "selected_domain":domain,
        "domain_pack":pack,
        "data_contract":contract,
        "probability_feature":config.get("probability_feature"),
        "variables":config.get("variables") or [],
        "weights":config.get("weights") or {},
        "matched_variables":config.get("matched_variables") or [],
        "governance":config.get("governance"),
        "blockers":blockers,
        "warnings":warnings,
        "ready":not blockers,
    }
