"""AgentOS authentication and usage tracking."""

from agentos.auth.models import User, UserStore, default_store
from agentos.auth.auth import (
    authenticate,
    create_user,
    generate_api_key,
    get_current_user,
    get_optional_user,
    get_user_by_email,
)
from agentos.auth.usage import UsageTracker, usage_tracker
from agentos.auth.usage_sqlite import UsageTrackerAsync, usage_tracker_async, UsageSummary
from agentos.auth.org_models import Organization, OrgMembership, Role, ApiKey
from agentos.auth.org_store import create_org, get_org, list_org_members, add_org_member, remove_org_member, get_api_key_info, check_scope, register_api_key

__all__ = [
    "User",
    "UserStore",
    "default_store",
    "authenticate",
    "create_user",
    "generate_api_key",
    "get_current_user",
    "get_optional_user",
    "get_user_by_email",
    "UsageTracker",
    "usage_tracker",
    "UsageTrackerAsync",
    "usage_tracker_async",
    "UsageSummary",
    "Organization",
    "OrgMembership",
    "Role",
    "ApiKey",
]

