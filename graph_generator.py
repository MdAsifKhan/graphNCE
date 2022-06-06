import build_graph

def graph_generator(width_basis=15, basis_type = "cycle", n_shapes = 5, shape_list=[[["house"]]], identifier = 'AA', add_edges = 0):
    ################################### EXAMPLE TO BUILD A SIMPLE REGULAR STRUCTURE ##########
    ## REGULAR STRUCTURE: the most simple structure:  basis + n small patterns of a single type
    ### 1. Choose the basis (cycle, torus or chain)
    ### 2. Add the shapes
    list_shapes = []
    for shape in shape_list:
        list_shapes += shape * n_shapes
    print(list_shapes)

    ### 3. Give a name to the graph
    name_graph = 'houses' + identifier
    sb.set_style('white')

    ### 4. Pass all these parameters to the Graph Structure
    G, communities, plugins, role_id = build_graph.build_structure(width_basis, basis_type, list_shapes, start=0,
                                                                   add_random_edges=add_edges,
                                                                   plot=True, savefig=False)
    return G, role_id