import jax
import jax.numpy as jnp
import jax.random as jr
import equinox as eqx
import equinox.nn as layers

class CNNBlock(eqx.Module):
    conv3x3 : layers.Conv2d
    lnorm1  : layers.LayerNorm
    conv1x1 : layers.Conv2d
    lnorm2  : layers.LayerNorm
    extrct1 : layers.Conv2d
    extrct2 : layers.Conv2d
    conv2_3x3: layers.Conv2d
    conv2_1x1: layers.Conv2d
    conv_skip: layers.Conv2d

    def __init__(
            self,
            input_size: int,
            out_c_one: int,
            out_c_two: int,
            key: jr.PRNGKey = jr.PRNGKey(42),
            dtype: jnp.dtype = jnp.bfloat16
    ):
        k1, k2, k3, k4, k5, k6, k7 = jr.split(key, 7)

        self.conv3x3 = layers.Conv2d( # Divides
            in_channels     = input_size,
            out_channels    = out_c_one,    # The information, usually x2 than input if stride 2
            kernel_size     = 3,            # How many close cells a cell can look in.
            stride          = 2,            # Downscale value, if 1 just extracts features.
            padding         = 1,
            dtype           = dtype,
            key             = k1
        )
        self.lnorm1 = layers.LayerNorm(
            shape = input_size,
            dtype = dtype
        )
        self.conv1x1 = layers.Conv2d(
            in_channels     = out_c_one,
            out_channels    = out_c_one,
            kernel_size     = 1,
            stride          = 1,
            padding         = 0,
            dtype           = dtype,
            key             = k2
        )
        self.lnorm2 = layers.LayerNorm(
            shape = out_c_one,
            dtype = dtype
        )
        self.extrct1    = layers.Conv2d(
            in_channels     = out_c_one,
            out_channels    = out_c_one,
            kernel_size     = 3,
            stride          = 1,
            padding         = 1,
            dtype           = dtype,
            key             = k3
        )
        self.conv2_3x3  = layers.Conv2d( # Divides
            in_channels     = out_c_one,
            out_channels    = out_c_two,
            kernel_size     = 3,
            stride          = 2,
            padding         = 1,
            dtype           = dtype,
            key             = k4
        )
        self.conv2_1x1  = layers.Conv2d(
            in_channels     = out_c_two,
            out_channels    = out_c_two,
            kernel_size     = 1,
            stride          = 1,
            padding         = 0,
            dtype           = dtype,
            key             = k5
        )
        self.extrct2    = layers.Conv2d(
            in_channels     = out_c_two,
            out_channels    = out_c_two,
            kernel_size     = 3,
            stride          = 1,
            padding         = 1,
            dtype           = dtype,
            key             = k6
        )
        self.conv_skip = layers.Conv2d(
            in_channels     = input_size,
            out_channels    = out_c_two,
            kernel_size     = 1,
            stride          = 4,
            dtype           = dtype,
            key             = k7
        )

    def __call__(
            self,
            x: jnp.ndarray
    ):
        x       = x.transpose(1, 2, 0) # Because shape is (c, h, w) but vmap likes (h, w, c) for some reason
        lx      = jax.vmap(jax.vmap(self.lnorm1, in_axes = 0), in_axes = 0)(x)
        lx      = lx.transpose(2, 0, 1) # And now we swap it back 
        c3x3    = self.conv3x3(lx)
        sil     = jax.nn.silu(c3x3)


        c1x1    = self.conv1x1(sil)
        sil2    = jax.nn.silu(c1x1)

        extr    = self.extrct1(sil2)
        sil3    = jax.nn.silu(extr)

        sil3    = sil3.transpose(1, 2, 0)
        lx2     = jax.vmap(jax.vmap(self.lnorm2, in_axes = 0), in_axes = 0)(sil3)
        lx2     = lx2.transpose(2, 0, 1)
        c3x3    = self.conv2_3x3(lx2)
        sil4    = jax.nn.silu(c3x3)

        # c1x1    = self.conv2_1x1(sil4)
        # extr    = self.extrct2(c1x1)
        # sil5    = jax.nn.silu(extr)

        skip    = self.conv_skip(lx)
        add     = sil4 + skip

        return add

class Bottleneck(eqx.Module):
    bottleneck: layers.Conv2d
    work_neck: layers.Conv2d
    exp_neck: layers.Conv2d

    def __init__(
            self,
            input_size: int,
            bottleneck_size: int,
            key: jr.PRNGKey = jr.PRNGKey(42),
            dtype: jnp.dtype = jnp.bfloat16
    ):
        k1, k2, k3 = jr.split(key, 3)
        self.bottleneck = layers.Conv2d(
            in_channels     = input_size,
            out_channels    = bottleneck_size,
            kernel_size     = 1,
            stride          = 1,
            padding         = 0,
            key             = k1,
            dtype           = dtype
        )
        self.work_neck = layers.Conv2d(
            in_channels     = bottleneck_size,
            out_channels    = bottleneck_size,
            kernel_size     = 3,
            stride          = 1,
            padding         = 1,
            key             = k2,
            dtype           = dtype
        )
        self.exp_neck = layers.Conv2d(
            in_channels     = bottleneck_size,
            out_channels    = input_size,
            kernel_size     = 1,
            stride          = 1,
            padding         = 0,
            key             = k3,
            dtype           = dtype
        )
    def __call__(
            self,
            x: jnp.ndarray
    ):
        reduce  = self.bottleneck(x)
        work    = self.work_neck(reduce)
        exp     = self.exp_neck(work)

        return exp + x

# class Solo(eqx.Module):
#     block: CNNBlock
#     bneck: Bottleneck
#     block2: CNNBlock
#     bneck2: Bottleneck
#     block3: CNNBlock
#     transition: layers.Conv2d
#     last_conv: layers.Conv2d

#     def __init__(
#             self,
#             input_size: int,
#             out_classes: int,
#             key: jr.PRNGKey = jr.PRNGKey(42),
#             dtype: jnp.dtype = jnp.bfloat16
#     ):
#         k1, k2, k3, k4, k5, k6, k7 = jr.split(key, 7)
#         self.block = CNNBlock(
#             input_size  = input_size,
#             out_c_one   = 32,
#             out_c_two   = 64,
#             key         = k1,
#             dtype       = dtype
#         )

#         self.bneck = Bottleneck(
#             input_size = 64,
#             bottleneck_size = 32,
#             key = k2,
#             dtype = dtype
#         )

#         self.block2 = CNNBlock(
#             input_size  = 64,
#             out_c_one   = 128,
#             out_c_two   = 256,
#             key         = k3,
#             dtype       = dtype
#         )

#         self.bneck2 = Bottleneck(
#             input_size = 256,
#             bottleneck_size = 128,
#             key = k4,
#             dtype = dtype
#         )

#         self.block3 = CNNBlock(
#             input_size  = 256,
#             out_c_one   = 512,
#             out_c_two   = 1024,
#             key         = k5,
#             dtype       = dtype
#         )
#         self.transition = layers.Conv2d(
#             in_channels     = 1024,
#             out_channels    = 256,
#             kernel_size     = 1,
#             stride          = 1,
#             padding         = 0,
#             key             = k6,
#             dtype           = dtype
#         )

#         self.last_conv = layers.Conv2d(
#             in_channels     = 256,
#             out_channels    = out_classes,
#             kernel_size     = 3,
#             stride          = 1,
#             padding         = 1,
#             key             = k7,
#             dtype           = dtype
#         )


#     def __call__(
#             self,
#             x: jnp.ndarray
#     ):
#         out     = eqx.filter_checkpoint(self.block)(x)
#         reduce  = self.bneck(out)
#         out2    = eqx.filter_checkpoint(self.block2)(reduce)
#         reduce2 = self.bneck2(out2)
#         out3    = eqx.filter_checkpoint(self.block3)(reduce2)

#         transition = self.transition(out3)
#         preds   = self.last_conv(transition)

#         return preds


class Solo(eqx.Module):
    block: CNNBlock
    bneck: Bottleneck
    block2: CNNBlock
    bneck2: Bottleneck
    block3: CNNBlock
    transition: layers.Conv2d
    last_conv: layers.Conv2d

    def __init__(
            self,
            input_size: int,
            out_classes: int,
            key: jr.PRNGKey = jr.PRNGKey(42),
            dtype: jnp.dtype = jnp.bfloat16
    ):
        k1, k2, k3, k4, k5, k6, k7 = jr.split(key, 7)
        self.block = CNNBlock(
            input_size  = input_size,
            out_c_one   = 64,
            out_c_two   = 128,
            key         = k1,
            dtype       = dtype
        )

        self.bneck = Bottleneck(
            input_size = 128,
            bottleneck_size = 64,
            key = k2,
            dtype = dtype
        )

        self.block2 = CNNBlock(
            input_size  = 128,
            out_c_one   = 256,
            out_c_two   = 512,
            key         = k3,
            dtype       = dtype
        )

        self.bneck2 = Bottleneck(
            input_size = 512,
            bottleneck_size = 128,
            key = k4,
            dtype = dtype
        )

        self.block3 = CNNBlock(
            input_size  = 512,
            out_c_one   = 512,
            out_c_two   = 512,
            key         = k5,
            dtype       = dtype
        )
        self.transition = layers.Conv2d(
            in_channels     = 512,
            out_channels    = 128,
            kernel_size     = 1,
            stride          = 1,
            padding         = 0,
            key             = k6,
            dtype           = dtype
        )

        self.last_conv = layers.Conv2d(
            in_channels     = 128,
            out_channels    = out_classes,
            kernel_size     = 3,
            stride          = 1,
            padding         = 1,
            key             = k7,
            dtype           = dtype
        )


    def __call__(
            self,
            x: jnp.ndarray
    ):
        out     = eqx.filter_checkpoint(self.block)(x)
        reduce  = self.bneck(out)
        out2    = eqx.filter_checkpoint(self.block2)(reduce)
        reduce2 = self.bneck2(out2)
        out3    = eqx.filter_checkpoint(self.block3)(reduce2)

        transition = self.transition(out3)
        preds   = self.last_conv(transition)

        return preds