"""Two-minute probe: does this account actually get a GPU and internet?

Both are gated behind Kaggle phone verification. A previous run requested both,
silently received neither, trained ViT on CPU with random weights, and was
cancelled after 20 hours. Never again without checking first.
"""
import torch, socket, sys

print("=" * 50)
print("GPU      available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU      name     :", torch.cuda.get_device_name(0))
    cc = torch.cuda.get_device_capability(0)
    print("GPU      capability:", cc, "(P100=6.0 breaks this torch; T4=7.5 ok)")
    x = torch.randn(64, 64, device="cuda") @ torch.randn(64, 64, device="cuda")
    print("GPU      matmul ok:", bool(torch.isfinite(x).all()))
else:
    print("!! NO GPU - account is probably not phone-verified")

try:
    socket.gethostbyname("huggingface.co")
    print("INTERNET dns      : OK")
    import timm
    m = timm.create_model("resnet18.a1_in1k", pretrained=True, num_classes=0)
    w = m.conv1.weight.detach()
    print("INTERNET weights  : DOWNLOADED  (conv1 std %.4f, non-random)" % w.std())
except Exception as e:
    print("!! NO INTERNET:", type(e).__name__, str(e)[:80])
print("=" * 50)
