"""Port planning for local multi-instance deployments."""
from __future__ import annotations


# Keep one reserved block per instance.  SGLang's DP-attention TCP endpoints
# are derived from the HTTP port, so adjacent HTTP ports are not independent.
DEPLOYMENT_PORT_STRIDE = 256


def deployment_service_ports(
    service_port: int,
    instance_count: int,
    *,
    stride: int = DEPLOYMENT_PORT_STRIDE,
) -> list[int]:
    """Return the HTTP port assigned to each local deployment instance."""
    if service_port < 1 or service_port > 65535:
        raise ValueError(f"service_port must be between 1 and 65535: {service_port}")
    if instance_count < 1:
        raise ValueError(f"instance_count must be positive: {instance_count}")
    if stride < 1:
        raise ValueError(f"service port stride must be positive: {stride}")
    last_reserved = service_port + (instance_count - 1) * stride + stride - 1
    if last_reserved > 65535:
        raise ValueError(
            "multi-instance service port plan exceeds TCP port range: "
            f"base={service_port} instances={instance_count} stride={stride}"
        )
    return [service_port + offset * stride for offset in range(instance_count)]
