"""Deploy the CPU web app using its server-only named Modal secret."""

from pathlib import Path
import modal

ROOT = Path(__file__).resolve().parents[1]
secret = modal.Secret.from_name("qwen-medical-web-config")
app = modal.App("qwen-peft-medical-web")
image = (modal.Image.debian_slim(python_version="3.12")
         .pip_install("fastapi==0.119.1", "authlib==1.6.5", "httpx==0.28.1", "itsdangerous==2.2.0")
         .add_local_dir(ROOT / "webapp", "/root/webapp", ignore=["__pycache__/"]))


@app.function(image=image, secrets=[secret], min_containers=0, max_containers=1,
              scaledown_window=300, timeout=1800)
@modal.concurrent(max_inputs=20)
@modal.asgi_app()
def web():
    from webapp.app import create_app
    return create_app()
