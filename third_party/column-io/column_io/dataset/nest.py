def _pack_nest_sequence_internal(schema, data_vec, comp, pos):
    if isinstance(schema, dict):
        output = {}
        for key in sorted(schema, key=comp):
            output[key], pos = _pack_nest_sequence_internal(
                schema[key], data_vec, comp, pos
            )
        return output, pos
    elif isinstance(schema, (list, tuple)):
        output = []
        for val in schema:
            new_val, pos = _pack_nest_sequence_internal(val, data_vec, comp, pos)
            output.append(new_val)
        return output, pos
    else:
        val = data_vec[pos]
        pos += 1
        return val, pos


def pack_nest_sequence(schema, data_vec: list, key=lambda x: x):
    # non nested data
    if nest_seq_leaf_num(schema) != len(data_vec):
        raise RuntimeError(
            (
                "schema and data_vec size mismatch"
                "schema : {}"
                "data vec: {}".format(schema, data_vec)
            )
        )
    if not isinstance(schema, (list, tuple, dict)):
        return data_vec[0]
    return _pack_nest_sequence_internal(schema, data_vec, key, 0)[0]


def nest_seq_leaf_num(schema, key=lambda x: x):
    ret = 0
    if isinstance(schema, dict):
        for key in sorted(schema, key=key):
            ret += nest_seq_leaf_num(schema[key], key)
    elif isinstance(schema, (list, tuple)):
        for val in schema:
            ret += nest_seq_leaf_num(val, key)
    else:
        ret += 1
    return ret
