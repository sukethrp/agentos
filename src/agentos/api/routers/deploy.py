from __future__ import annotations
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from agentos.deploy.k8s_deployer import (
    deploy_agent,
    scale_agent,
    teardown_agent,
    KubernetesConfig,
    AgentDeployConfig,
)

router = APIRouter(prefix="/deploy", tags=["deploy"])


def _k8s_config(
    kubeconfig_path: str | None = None,
    namespace: str = "default",
    image_registry: str = "ghcr.io",
    image_tag: str = "latest",
) -> KubernetesConfig:
    return KubernetesConfig(
        kubeconfig_path=kubeconfig_path,
        namespace=namespace,
        image_registry=image_registry,
        image_tag=image_tag,
    )


class DeployAgentBody(BaseModel):
    name: str
    resources: dict[str, str] = Field(
        default_factory=lambda: {"cpu": "100m", "memory": "128Mi"}
    )
    replicas: int = 1
    env_vars: dict[str, str] = Field(default_factory=dict)
    kubeconfig_path: str | None = None
    namespace: str = "default"
    image_registry: str = "ghcr.io"
    image_tag: str = "latest"


class ScaleBody(BaseModel):
    replicas: int


@router.post("/k8s")
async def deploy_k8s(body: DeployAgentBody) -> dict:
    try:
        cfg = AgentDeployConfig(
            name=body.name,
            resources=body.resources,
            replicas=body.replicas,
            env_vars=body.env_vars,
        )
        k8s_cfg = _k8s_config(
            body.kubeconfig_path,
            body.namespace,
            body.image_registry,
            body.image_tag,
        )
        return await deploy_agent(cfg, k8s_cfg)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/k8s/{agent_name}/scale")
async def deploy_k8s_scale(
    agent_name: str,
    body: ScaleBody,
    kubeconfig_path: str | None = None,
    namespace: str = "default",
    image_registry: str = "ghcr.io",
    image_tag: str = "latest",
) -> dict:
    try:
        k8s_cfg = _k8s_config(kubeconfig_path, namespace, image_registry, image_tag)
        await scale_agent(agent_name, body.replicas, k8s_cfg)
        return {"status": "scaled", "agent_name": agent_name, "replicas": body.replicas}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/k8s/{agent_name}")
async def deploy_k8s_teardown(
    agent_name: str,
    kubeconfig_path: str | None = None,
    namespace: str = "default",
    image_registry: str = "ghcr.io",
    image_tag: str = "latest",
) -> dict:
    try:
        k8s_cfg = _k8s_config(kubeconfig_path, namespace, image_registry, image_tag)
        await teardown_agent(agent_name, k8s_cfg)
        return {"status": "teardown", "agent_name": agent_name}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
