import shutil
from pathlib import Path

def deploy_user_config():
    config_dir = Path.home() / ".config" / "shell_gpt"
    config_dir.mkdir(parents=True, exist_ok=True)

    pkg_dir = Path(__file__).resolve().parent
    roles_src = pkg_dir / "roles"
    funcs_src = pkg_dir / "functions"

    # 强制覆盖角色
    dest_roles = config_dir / "roles"
    if dest_roles.exists():
        shutil.rmtree(dest_roles)
    if roles_src.exists():
        shutil.copytree(roles_src, dest_roles)

    # 强制覆盖函数
    dest_functions = config_dir / "functions"
    if dest_functions.exists():
        shutil.rmtree(dest_functions)
    if funcs_src.exists():
        shutil.copytree(funcs_src, dest_functions)

    # 不再生成 .sgptrc，交给程序自动提示输入 API Key。
    # 首次运行时会要求输入，并使用 config.py 中的 DeepSeek 默认值。