from __future__ import annotations

from dataclasses import dataclass

from .analyzer import AnalysisResult


@dataclass(frozen=True, slots=True)
class MixRecommendation:
    title: str
    detail: str
    priority: str = "medium"


def recommend(result: AnalysisResult) -> list[MixRecommendation]:
    recommendations: list[MixRecommendation] = []

    if result.peak_dbfs > -1.0:
        recommendations.append(
            MixRecommendation(
                title="Reduce peak level",
                detail="Leave more headroom before downstream processing or summing.",
                priority="high",
            )
        )

    if result.crest_factor_db < 6.0:
        recommendations.append(
            MixRecommendation(
                title="Check dynamics",
                detail="Low crest factor can indicate dense compression or limiting; verify transients before adding more processing.",
                priority="medium",
            )
        )

    if not recommendations:
        recommendations.append(
            MixRecommendation(
                title="No immediate level warning",
                detail="This basic pass found no urgent peak/dynamic issue. Deeper spectral and stereo analysis will follow.",
                priority="low",
            )
        )

    return recommendations
