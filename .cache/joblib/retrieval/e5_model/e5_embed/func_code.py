# first line: 116
@memory.cache
def e5_embed(text_list: list[str]):
    e5 = E5()
    res = e5(text_list)
    return res
