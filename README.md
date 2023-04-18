# Embedded Geometry

This is the supporting repository to our paper *Geometry in global coordinates with
applications in optimal transport and
mechanics*.

The main formulas involves derivatives of a number of operators, including projection $\Pi$, metric $\mathsf{g}$, and Christoffel function $\Gamma$. We use [Jax](https://jax.readthedocs.io/en/latest/index.html) for numerical derivative, in particular, [jvp](https://jax.readthedocs.io/en/latest/_autosummary/jax.jvp.html) is the main tool for directional derivative. The jax functions [grad](https://jax.readthedocs.io/en/latest/_autosummary/jax.grad.html#jax.grad) and [jacfwd](https://jax.readthedocs.io/en/latest/_autosummary/jax.jacfwd.html) are also useful.

Examples verifies properties of the Christoffel function, the curvature formulas and Bianchi identities.

In [JaxRigidBodyDynamics.ipynb](https://github.com/dnguyend/EmbeddedGeometry/blob/main/colab/JaxRigidBodyDynamics.ipynb) we show examples for rigid body mechanics, and in [KMCTests.ipynb](https://github.com/dnguyend/EmbeddedGeometry/blob/main/colab/KMCTests.ipynb) we show the numerical experiments for the Kim-McCann metric.
