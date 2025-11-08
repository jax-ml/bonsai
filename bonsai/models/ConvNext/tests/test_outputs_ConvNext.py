import jax 
import jax.numpy as jnp 
import  torch 
from absl.testing import absltest
from huggingface_hub import snapshot_download
from transformers import ConvNextForImageClassification


from bonsai.models.ConvNext import params
from bonsai.models.ConvNext import modeling as model_lib


class TestModuleForwardPasses(absltest.TestCase):

    def setUp(self):
        super().setUp()
        model_name = "facebook/convnext-large-224"
        model_ckpt_path = snapshot_download(model_name)
        self.bonsai_model = params._create_convnext_from_pretrained(
            model_lib.ConvNeXt
            ,model_ckpt_path)
        self.baseline_model = ConvNextForImageClassification.from_pretrained(model_name)
        self.bonsai_model.eval()
        self.baseline_model.eval()

        self.batch_size = 32 
        self.image_shape = (self.batch_size,224,224,3)


    def test_embeddings(self):
        torch_emb = self.baseline_model.convnext.embeddings 
        nnx_emb = self.bonsai_model.embeddings  

        jx = jax.random.normal(jax.random.key(0), self.image_shape, dtype=jnp.float32)
        tx = torch.tensor(jx).permute(0, 3, 1, 2)

        with torch.no_grad():
            ty = torch_emb(tx) 
        jy = nnx_emb(jx) 

        torch.testing.assert_close(torch.tensor(jy), ty, rtol=1e-5, atol=1e-5)


    def test_full(self):
        jx = jax.random.normal(jax.random.key(0), self.image_shape , dtype = jnp.float32)
        tx = torch.tensor(jx).permute(0,3,1,2)


        with torch.no_grad():
            ty = self.baseline_model(tx).logits
        jy = self.bonsai_model(jx)



if __name__ == "__main__":
    absltest.main()