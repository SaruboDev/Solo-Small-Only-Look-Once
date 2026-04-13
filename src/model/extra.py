import os
import jax
import jax.numpy as jnp
import jax.random as jr
import equinox as eqx
import math

def save_model(path: str, model, opt_state, step, train_key: jr.PRNGKey, epoch = 0):
    """
    Saves a model at the current step and state.
    """
    saved = {
        "model"     : model,
        "opt_state" : opt_state,
        "step"      : step,
        "epoch"     : epoch,
        "train_key" : train_key
    }
    output_dir = os.path.dirname(path)

    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok = True)

    tmp_path = path + ".tmp"

    eqx.tree_serialise_leaves(tmp_path, saved)
    os.rename(tmp_path, path)

def load_model(path: str, model_base, optimizer, is_inexact_init: bool = True):
    """
    Loads and deserializes a model checkpoint.

    :params is_inexact_init: How the user initialized the optimizer, if True will use "eqx.filter(model, eqx.is_inexact_array)".
    :type is_inexact_init: bool
    """

    if is_inexact_init:
        opt_state = optimizer.init(eqx.filter(model_base, eqx.is_inexact_array))
    else:
        opt_state = optimizer.init(model_base)

    load = {
        "model"     : model_base,
        "opt_state" : opt_state,
        "step"      : 0,
        "epoch"     : 0,
        "train_key" : jr.PRNGKey(42)
    }

    loaded = eqx.tree_deserialise_leaves(path, load)
    return loaded["model"], loaded["opt_state"], loaded["step"], loaded["epoch"], loaded["train_key"]

def summary(model):
    """
    Gives a summary of the initialized model.
    """
    def __memory(params):
        convert = {
            jnp.int8 : 1,
            jnp.int16 : 2,
            jnp.int32 : 4,
            jnp.int64 : 8,
            jnp.bfloat16 : 2,
            jnp.float16 : 2,
            jnp.float32 : 4,
            jnp.float64 : 8,
            jnp.bool_ : 1,
        }

        total = 0

        for leaf in jax.tree_util.tree_leaves(params):
            if isinstance(leaf, jnp.ndarray):
                bSize   = convert.get(leaf.dtype, 4)
                total   += leaf.size * bSize
        return total

    # Calculates every params
    params              = eqx.filter(model, eqx.is_array)
    trainable_params    = eqx.filter(model, eqx.is_inexact_array)
    param_count         = sum(x.size for x in jax.tree_util.tree_leaves(params))
    trainable_count     = sum(x.size for x in jax.tree_util.tree_leaves(trainable_params))

    total_size          = __memory(params)
    trainable_size      = __memory(trainable_params)
    nonTrainable_size   = total_size - trainable_size

    def __convert_size(size_bytes):
        if size_bytes == 0 or size_bytes == None:
            return "0B"
        size_name = ("B", "KB", "MB", "GB", "TB", "PB", "EB", "ZB", "YB")
        i = int(math.floor(math.log(size_bytes, 1024)))
        p = math.pow(1024, i)
        s = round(size_bytes / p, 2)
        return "%s %s" % (s, size_name[i])

    # We finally print the Total Parameters, Trainable Parameters and Non-Trainable Parameters
    print(f"Total Params: {"{:,}".format(param_count)} - {__convert_size(total_size)}")
    print(f"Trainable Params: {"{:,}".format(trainable_count)} - {__convert_size(trainable_size)}")
    print(f"Non-trainable Params: {"{:,}".format(param_count - trainable_count)} - {__convert_size(nonTrainable_size)}")
