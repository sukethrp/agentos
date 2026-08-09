"""AgentOS authentication and usage tracking."""

from agentos.auth.auth import (
    authenticate,
    create_user,
    generate_api_key,
    get_current_user,
    get_optional_user,
    get_user_by_email,
)
from agentos.auth.models import User, UserStore, default_store
from agentos.auth.org_models import ApiKey, Organization, OrgMembership, Role
from agentos.auth.org_store import (
    add_org_member,
    check_scope,
    create_org,
    get_api_key_info,
    get_org,
    list_org_members,
    register_api_key,
    remove_org_member,
)
from agentos.auth.usage import UsageTracker, usage_tracker
from agentos.auth.usage_sqlite import (
    UsageSummary,
    UsageTrackerAsync,
    usage_tracker_async,
)

__all__ = [
    "ApiKey",
    "OrgMembership",
    "Organization",
    "Role",
    "UsageSummary",
    "UsageTracker",
    "UsageTrackerAsync",
    "User",
    "UserStore",
    "add_org_member",
    "authenticate",
    "check_scope",
    "create_org",
    "create_user",
    "default_store",
    "generate_api_key",
    "get_api_key_info",
    "get_current_user",
    "get_optional_user",
    "get_org",
    "get_user_by_email",
    "list_org_members",
    "register_api_key",
    "remove_org_member",
    "usage_tracker",
    "usage_tracker_async",
]
