# Embedded Geometry

This is the supporting repository to our paper *Geometry in global coordinates with
applications in optimal transport and
mechanics*.

## The derivation of the Christoffel function for the reflector antenna cost on an optimal transport problem on fixed-rank semidefinite matrices is given [here](https://github.com/dnguyend/EmbeddedGeometry/blob/main/derive_gamma.pdf).

The main formulas involves derivatives of a number of operators, including projection $\Pi$, metric $\mathsf{g}$, and Christoffel function $\Gamma$. We use [Jax](https://jax.readthedocs.io/en/latest/index.html) for numerical derivative, in particular, [jvp](https://jax.readthedocs.io/en/latest/_autosummary/jax.jvp.html) is the main tool for directional derivative. The jax functions [grad](https://jax.readthedocs.io/en/latest/_autosummary/jax.grad.html#jax.grad) and [jacfwd](https://jax.readthedocs.io/en/latest/_autosummary/jax.jacfwd.html) are also useful.

Examples verifies properties of the Christoffel function, the curvature formulas and Bianchi identities.

In [JaxRigidBodyDynamics.ipynb](https://github.com/dnguyend/EmbeddedGeometry/blob/main/colab/JaxRigidBodyDynamics.ipynb) we show examples for rigid body mechanics.

In [SemidefiniteOptimalTransportAntennaCost.ipynb](https://github.com/dnguyend/EmbeddedGeometry/blob/main/colab/SemidefiniteOptimalTransportAntennaCost.ipynb) we show numerical verifications of the Christoffel function and the cross curvature for an optimal transport problem with reflector antenna cost.

In [KMCTests.ipynb](https://github.com/dnguyend/EmbeddedGeometry/blob/main/colab/KMCTests.ipynb) we show a few other numerical experiments for the Kim-McCann metric.
