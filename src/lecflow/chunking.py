
def group_adjacent(sentences: list[str], cluster_labels: list[int]) -> list[list[str]]:
    '''Groups consecutive sentences of the same cluster into ordered blocks. Maintains 
    sentence order; topic mentioned in two different spots becomes two different blocks.'''
    if not sentences: #avoid error if empty
        return [] 

    current_cluster = cluster_labels[0]
    current_block, grouped_blocks = [], []
    for idx, sentence in enumerate(sentences):
        if cluster_labels[idx] == current_cluster:
            current_block.append(sentence)
        else:
            grouped_blocks.append(current_block)
            current_cluster = cluster_labels[idx]
            current_block = [sentence]
    grouped_blocks.append(current_block)
    return grouped_blocks
    

