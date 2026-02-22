from __future__ import annotations
from typing import Any
from pydantic import BaseModel, Field
from agentos.core.types import AgentConfig


class KubernetesConfig(BaseModel):
    kubeconfig_path: str | None = None
    namespace: str = "default"
    image_registry: str = "ghcr.io"
    image_tag: str = "latest"


class AgentDeployConfig(BaseModel):
    name: str
    resources: dict[str, str] = Field(default_factory=lambda: {"cpu": "100m", "memory": "128Mi"})
    replicas: int = 1
    env_vars: dict[str, str] = Field(default_factory=dict)


async def deploy_agent(
    agent_config: AgentConfig | AgentDeployConfig,
    k8s_config: KubernetesConfig,
) -> dict[str, Any]:
    if isinstance(agent_config, AgentConfig):
        deploy_cfg = AgentDeployConfig(
            name=agent_config.name,
            resources={"cpu": "100m", "memory": "128Mi"},
            replicas=1,
            env_vars={
                "AGENTOS_MODEL": agent_config.model,
                "AGENTOS_SYSTEM_PROMPT": agent_config.system_prompt,
            },
        )
    else:
        deploy_cfg = agent_config
    from kubernetes_asyncio import client, config
    if k8s_config.kubeconfig_path:
        await config.load_kube_config(config_file=k8s_config.kubeconfig_path)
    else:
        await config.load_kube_config()
    ns = k8s_config.namespace
    name = deploy_cfg.name
    image = f"{k8s_config.image_registry}/agentos/agent:{k8s_config.image_tag}"
    cpu = deploy_cfg.resources.get("cpu", "100m")
    mem = deploy_cfg.resources.get("memory", "128Mi")
    env_list = [
        client.V1EnvVar(name=k, value=v)
        for k, v in deploy_cfg.env_vars.items()
    ]
    container = client.V1Container(
        name=name,
        image=image,
        env=env_list,
        resources=client.V1ResourceRequirements(
            requests={"cpu": cpu, "memory": mem},
            limits={"cpu": cpu, "memory": mem},
        ),
        ports=[client.V1ContainerPort(container_port=8000)],
    )
    pod_spec = client.V1PodSpec(
        containers=[container],
        restart_policy="Always",
    )
    pod_template = client.V1PodTemplateSpec(
        metadata=client.V1ObjectMeta(labels={"app": name}),
        spec=pod_spec,
    )
    deployment = client.V1Deployment(
        api_version="apps/v1",
        kind="Deployment",
        metadata=client.V1ObjectMeta(name=name),
        spec=client.V1DeploymentSpec(
            replicas=deploy_cfg.replicas,
            selector=client.V1LabelSelector(match_labels={"app": name}),
            template=pod_template,
        ),
    )
    service = client.V1Service(
        api_version="v1",
        kind="Service",
        metadata=client.V1ObjectMeta(name=name),
        spec=client.V1ServiceSpec(
            selector={"app": name},
            ports=[client.V1ServicePort(port=8000, target_port=8000)],
        ),
    )
    configmap = client.V1ConfigMap(
        api_version="v1",
        kind="ConfigMap",
        metadata=client.V1ObjectMeta(name=f"{name}-config"),
        data=deploy_cfg.env_vars,
    )
    hpa = client.V2HorizontalPodAutoscaler(
        api_version="autoscaling/v2",
        kind="HorizontalPodAutoscaler",
        metadata=client.V1ObjectMeta(name=name),
        spec=client.V2HorizontalPodAutoscalerSpec(
            scale_target_ref=client.V2CrossVersionObjectReference(
                api_version="apps/v1",
                kind="Deployment",
                name=name,
            ),
            min_replicas=1,
            max_replicas=max(deploy_cfg.replicas * 2, 10),
            metrics=[
                client.V2MetricSpec(
                    type="Resource",
                    resource=client.V2ResourceMetricSource(
                        name="cpu",
                        target=client.V2MetricTarget(
                            type="Utilization",
                            average_utilization=70,
                        ),
                    ),
                ),
            ],
        ),
    )
    async with client.ApiClient() as api:
        apps = client.AppsV1Api(api)
        core = client.CoreV1Api(api)
        autoscaling = client.AutoscalingV2Api(api)
        await apps.create_namespaced_deployment(namespace=ns, body=deployment)
        await core.create_namespaced_service(namespace=ns, body=service)
        await core.create_namespaced_config_map(namespace=ns, body=configmap)
        await autoscaling.create_namespaced_horizontal_pod_autoscaler(namespace=ns, body=hpa)
    return {
        "deployment": name,
        "service": name,
        "configmap": f"{name}-config",
        "hpa": name,
        "namespace": ns,
    }


async def scale_agent(agent_id: str, replicas: int, k8s_config: KubernetesConfig) -> None:
    from kubernetes_asyncio import client, config
    if k8s_config.kubeconfig_path:
        await config.load_kube_config(config_file=k8s_config.kubeconfig_path)
    else:
        await config.load_kube_config()
    ns = k8s_config.namespace
    async with client.ApiClient() as api:
        apps = client.AppsV1Api(api)
        await apps.patch_namespaced_deployment_scale(
            name=agent_id,
            namespace=ns,
            body={"spec": {"replicas": replicas}},
        )


async def teardown_agent(agent_id: str, k8s_config: KubernetesConfig) -> None:
    from kubernetes_asyncio import client, config
    if k8s_config.kubeconfig_path:
        await config.load_kube_config(config_file=k8s_config.kubeconfig_path)
    else:
        await config.load_kube_config()
    ns = k8s_config.namespace
    delete_opts = client.V1DeleteOptions(propagation_policy="Foreground")
    async with client.ApiClient() as api:
        apps = client.AppsV1Api(api)
        core = client.CoreV1Api(api)
        autoscaling = client.AutoscalingV2Api(api)
        try:
            await apps.delete_namespaced_deployment(name=agent_id, namespace=ns, body=delete_opts)
        except client.rest.ApiException:
            pass
        try:
            await core.delete_namespaced_service(name=agent_id, namespace=ns, body=delete_opts)
        except client.rest.ApiException:
            pass
        try:
            await core.delete_namespaced_config_map(name=f"{agent_id}-config", namespace=ns, body=delete_opts)
        except client.rest.ApiException:
            pass
        try:
            await autoscaling.delete_namespaced_horizontal_pod_autoscaler(name=agent_id, namespace=ns, body=delete_opts)
        except client.rest.ApiException:
            pass
