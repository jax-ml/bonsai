# Copyright 2025 The JAX Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os

import numpy as np
import torch

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

from mamba_ssm import MambaLMHeadModel

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "artifacts")


def main():
    model = MambaLMHeadModel.from_pretrained(
        "state-spaces/mamba2-130m",
        device="cuda",
        dtype=torch.float32,
    ).eval()

    for mod in model.modules():
        if hasattr(mod, "use_mem_eff_path"):
            mod.use_mem_eff_path = False

    input_ids = np.array([[1, 2, 3, 4, 5, 6, 7, 8]], dtype=np.int32)
    torch_input = torch.tensor(input_ids, device="cuda")

    with torch.no_grad():
        hidden = model.backbone(torch_input).cpu().numpy()
        logits = model.lm_head(model.backbone(torch_input)).cpu().numpy()

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    np.savez_compressed(
        os.path.join(OUTPUT_DIR, "golden_mamba2_130m.npz"),
        input_ids=input_ids,
        last_hidden_state=hidden.astype(np.float32),
        logits_slice=logits[:, :, :256].astype(np.float32),
    )
    print(f"Saved to {OUTPUT_DIR}/golden_mamba2_130m.npz")


if __name__ == "__main__":
    main()
