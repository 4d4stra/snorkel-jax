import jax.numpy as jnp
import jax

def _loss_inv_Z(Z,O_inv,mask):
    return jnp.linalg.norm(jnp.where(mask,O_inv + Z @ Z.T,0)) ** 2

Zloss=jax.jit(_loss_inv_Z)
grad_Zloss = jax.jit(jax.grad(_loss_inv_Z))

def _loss_inv_mu(mu,Q,P,O,mask):
    loss_1 = jnp.linalg.norm(Q - mu @ P @ mu.T) ** 2
    loss_2 = jnp.linalg.norm(jnp.sum(mu @ P, axis=1) - jnp.diag(O)) ** 2
    return loss_1 + loss_2

invMUloss=jax.jit(_loss_inv_mu)
grad_invMUloss = jax.jit(jax.grad(_loss_inv_mu))

def _loss_mu(mu,O,P,mask):
    loss_1 = jnp.linalg.norm(jnp.where(mask,O - mu @ P @ mu.T , 0))**2
    loss_2 = jnp.linalg.norm(jnp.sum(mu @ P,axis = 1) - jnp.diag(O)) ** 2
    loss=loss_1 + loss_2 #+ self._loss_l2(l2=l2)
    return loss
    
MUloss=jax.jit(_loss_mu)
grad_MUloss = jax.jit(jax.grad(_loss_mu))