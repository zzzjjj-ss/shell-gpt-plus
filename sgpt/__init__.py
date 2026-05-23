from .init_config import deploy_user_config
deploy_user_config()

from .app import main as main
from .app import entry_point as cli  # noqa: F401