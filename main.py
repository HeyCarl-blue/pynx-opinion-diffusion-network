import networkx as nx
import pandas as pd
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from PIL import Image, ImageTk
import numpy as np

import csv
import random
import string
import os
from datetime import datetime

import PySimpleGUI as gui

def random_str(existing_keys):
    l = random.randint(3, 10)
    s = ''
    while True:
        s = ''.join(random.choice(string.ascii_lowercase) for _ in range(l))
        if s not in existing_keys:
            return s



def load_csv_graph(filepath):
    r = pd.read_csv(filepath, header=None)

    if r.shape[0] == r.shape[1]:
        return (nx.from_pandas_adjacency(r), {}, {})
    elif r.shape[1] == r.shape[0] + 1:
        opinions = {}
        node_opinions = {}
        for o in np.unique(r.iloc[:, -1]):
            opinions[o] = random.random()
        for i in range(len(r)):
            node_opinions[i] = r[len(r.columns)-1][i]
        r.drop(columns=r.columns[-1], axis=1, inplace=True)
        return (nx.from_pandas_adjacency(r), opinions, node_opinions)
    elif r.shape[1] == r.shape[0] + 2:
        opinions = {}
        node_opinions = {}
        for i in range(len(r)):
            o = r[len(r.columns)-2][i]
            v = r[len(r.columns)-1][i]
            node_opinions[i] = o
            opinions[o] = v
        r.drop(columns=r.columns[-2], axis=1, inplace=True)
        r.drop(columns=r.columns[-1], axis=1, inplace=True)
        return(nx.from_pandas_adjacency(r), opinions, node_opinions)

    else:
        raise RuntimeError('csv not formatted correctly')


def get_graph_tk_image(g, colors=None, node_opinions=None):
    fig, ax = plt.subplots()

    pos = nx.spring_layout(g, seed=2023)
    if colors is None:
        nx.draw(g, pos=pos, ax=ax, node_size=80, edge_color='#00000022')
    else:
        color_map = []
        for node in g:
            color_map.append(colors[node_opinions[node][0]])
        nx.draw(g, pos=pos, ax=ax, node_color=color_map, node_size=80, edge_color='#00000022')
    fig.canvas.draw()
    data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    
    pil_image = Image.fromarray(data).convert('RGB')

    return (ImageTk.PhotoImage(pil_image), fig.canvas.get_width_height())


def randomize_node_opinions(g, opinions):
    node_opinions = {}
    for node in g:
        node_opinions[node] = random.choice(list(opinions.keys()))
    return node_opinions


def generate_opinion_colors(opinions):
    colors = {}
    for o in opinions.keys():
        r = lambda: random.randint(0, 255)
        color = '#{:02X}{:02X}{:02X}'.format(r(), r(), r())
        colors[o] = color
    return colors



g = None

opinions = {
    "a": 0.5,
    "b": 0.1,
    "c": 0.7,
    "d": 0.9
}

node_opinions = {}

t_stay = 5

node_opinions_curr = {}

opinion_changes = {}

opinion_colors = {
    "a": "#ff0000",
    "b": "#00ff00",
    "c": "#0000ff",
    "d": "#ff00ff"
}

started = False

curr_step = 0

graph_origin = ""
graph_origin_info = ""

gui.theme('SystemDefaultForReal')
layout = [
    [gui.Menu([
        ['File', ['Load Graph', 'Save Graph as csv']],
        ['Edit', ['Opinions', 'Nodes']],
        ['Tools', ['Generate random graph']]
    ])],
    [gui.Image(key='-MAIN IMAGE-', expand_x=True, expand_y=True)],
    [gui.Text('t_stay'), gui.InputText(str(t_stay), size=5, key="-T_STAY-"), gui.HorizontalSeparator(), gui.Text('⏹️', text_color='red', key='-STARTED-'), gui.Button('Start'), gui.Button('Colors'), gui.Button('Stop'), gui.HorizontalSeparator(), gui.Text("step: 0", key="-STEPTEXT-"), gui.InputText(1, size=4, key="-STEPSKIP-"), gui.Button('>>', key="-NEXT-")]
    ]

window = gui.Window('Opinion Diffusion', layout=layout, resizable=True, size=(800, 700))

while True:
    event, values = window.read()
    print(event, values)

    if event == 'Start':
        print(node_opinions)
        if g is None:
            gui.popup_cancel("Graph doesn't exists")
            continue
        op = list(opinions.keys())
        correct = True
        for (k, v) in node_opinions.items():
            if v not in op:
                gui.popup_cancel("Node '" + str(k) + "' has opinion '" + str(v) + "' which isn't specified as an opinion")
                correct = False
                break
        if not correct:
            continue

        started = True
        window['-STARTED-'].update(value='▶️', text_color='green')
        t_stay = int(values['-T_STAY-'])
        curr_step = 0
        window['-STEPTEXT-'].update(value="step: " + str(curr_step))
        node_opinions_curr = {}
        opinion_changes = {}
        for (k, v) in node_opinions.items():
            node_opinions_curr[k] = (v, 0)
            opinion_changes[k] = []
        print(node_opinions_curr)
        opinion_colors = generate_opinion_colors(opinions)
        print(opinion_colors)

        (tk_image, size) = get_graph_tk_image(g, opinion_colors, node_opinions_curr)
        window['-MAIN IMAGE-'].update(data=tk_image, size=size)
    

    if started and event == 'Colors':
        colors_window = gui.Window('Colors', layout=[[
            gui.Column(layout=[
                    [
                        gui.Text("⬛", text_color=opinion_colors[o], auto_size_text=True), gui.Text(o), gui.Text(opinions[o])
                    ] for o in opinions ], scrollable=True, vertical_scroll_only=True, expand_x=True, expand_y=True)
        ]], finalize=True, size=(400, 450), resizable=True)

    

    if started and event == '-NEXT-':
        step_skip = int(values['-STEPSKIP-'])
        if step_skip < 1:
            step_skip = 1

        update_image = False

        for _ in range(step_skip):
            curr_step += 1
            temp_opinions = node_opinions_curr.copy()

            for node in g:
                opinion, last_update = node_opinions_curr[node]
                opinion_won = random.random() <= opinions[opinion]

                if last_update < t_stay or opinion_won:
                    temp_opinions[node] = (opinion, last_update + 1)
                    continue

                neighbors = [n for n in g[node]]

                if neighbors == []:
                    continue

                n = random.choice(neighbors)
                winning_opinion, _ = node_opinions_curr[n]
                if winning_opinion != opinion:
                    update_image = True
                    temp_opinions[node] = (winning_opinion, 0)
                    opinion_changes[node].append(n)
                else:
                    temp_opinions[node] = (winning_opinion, int(t_stay / 2))
                    opinion_changes[node].append(n)

            node_opinions_curr = temp_opinions
        
        # Update
        window['-STEPTEXT-'].update(value="step: " + str(curr_step))

        if update_image:
            (tk_image, size) = get_graph_tk_image(g, opinion_colors, node_opinions_curr)
            window['-MAIN IMAGE-'].update(data=tk_image, size=size)
    

    if started and event == 'Stop':
        started = False
        window['-STARTED-'].update(value='⏹️', text_color='red')
        folder = datetime.now().strftime("%d-%m-%Y")
        folder_time = datetime.now().strftime("%H_%M_%S")
        folder_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'logs', folder, folder_time)

        try:
            os.makedirs(folder_path)
        except FileExistsError:
            pass

        log_path = os.path.join(folder_path, 'graph_info.log')

        degrees = sorted(list(g.degree(g)), key=lambda x: x[1])
        min_deg = list(filter(lambda x: x[1] == degrees[0][1], degrees))
        max_deg = list(filter(lambda x: x[1] == degrees[-1][1], degrees))

        with open(log_path, 'w+') as file:
            file.write(graph_origin + "\n")
            file.write(graph_origin_info + "\n")
            file.write("\n")

            file.write("Graph Informations:\n")
            file.write("\tNodes: " + str(len(g)) + "\n")
            file.write("\tEdges: " + str(len(g.edges)) + "\n")
            file.write("\tNodes with maximum degree: " + str(list(map(lambda x: x[0], max_deg))) + " (" + str(degrees[-1][1]) + ")\n")
            file.write("\tNodes with minimum degree: " + str(list(map(lambda x: x[0], min_deg))) + " (" + str(degrees[0][1]) + ")\n")
            try:
                file.write("\tCenters: " + str(nx.center(g)) + " (" + str(len(g[nx.center(g)[0]])) +")" + "\n")
                file.write("\tRadius: " + str(nx.radius(g)) + "\n")
                file.write("\tDiameter: " + str(nx.diameter(g)) + "\n")
            except nx.exception.NetworkXError:
                file.write("\tGraph not connected")
            file.write("\n")

            file.write("t_stay: " + str(t_stay) + "\n\n")

            file.write("Opinions: " + str(len(opinions)))
            for (op, res) in sorted(opinions.items(), key=lambda x:x[1], reverse=True):
                file.write("\n\t" + str(op) + "\t--\t" + str(res))
            
            file.write("\n\nSimulation steps: " + str(curr_step) + "\n")
            
            file.write("\nSimulation start:\n")
            file.write("\tOpinions:")
            for (op, res) in sorted(opinions.items(), key=lambda x:x[1], reverse=True):
                file.write("\n\t\tnodes with opinion " + str(op) + "(" + str(res) + ")" + ": " + str(len(list(filter(lambda x: node_opinions[x] == op, node_opinions)))))
            
            file.write("\n\nSimulation end:\n")
            file.write("\tOpinions:")
            for (op, res) in sorted(opinions.items(), key=lambda x:x[1], reverse=True):
                file.write("\n\t\tnodes with opinion " + str(op) + "(" + str(res) + ")" + ": " + str(len(list(filter(lambda x: node_opinions_curr[x][0] == op, node_opinions_curr)))))

            with open(os.path.join(folder_path, 'nodes_info.csv'), 'w') as csv_file:
                writer = csv.writer(csv_file)

                top = {}
                for node in g:
                    top[node] = [0,0]
                
                for node in g:
                    top[node][0] = len(opinion_changes[node])
                    for c in opinion_changes[node]:
                        top[c][1] += 1
                
                betweeness = nx.betweenness_centrality(g)
                
                writer.writerow(["Node", "Times it made someone change opinion", "Times it changed opinion", "Degree", "Betweenness Centrality", "Closeness Centrality", "Clustering coefficient"])
                for (k, v) in sorted(top.items(), key=lambda x:x[1][1], reverse=True):
                    writer.writerow([k, v[1], v[0], g.degree(k), betweeness[k], nx.closeness_centrality(g, k), nx.clustering(g, k)])
        
        img_start_file = os.path.join(folder_path, 'img_start.png')
        img_end_file = os.path.join(folder_path, 'img_end.png')

        fig = plt.figure(figsize = (30, 30))
        legend_elements = [
            Line2D([0], [0], marker='o', color='w', label=str(k), markerfacecolor=v, markersize=28)
            for (k, v) in opinion_colors.items()
        ]
        ax = plt.subplot(111)
        ax.legend(handles=legend_elements, fontsize=28)
        pos = nx.spring_layout(g, seed=2023)

        color_map = []
        for node in g:
            color_map.append(opinion_colors[node_opinions[node]])

        nx.draw(g, pos=pos, node_size=3000, node_color=color_map, with_labels=True, font_size=26)
        plt.savefig(img_start_file, format='PNG')

        color_map = []
        for node in g:
            color_map.append(opinion_colors[node_opinions_curr[node][0]])

        nx.draw(g, pos=pos, node_size=3000, node_color=color_map, with_labels=True, font_size=26)
        plt.title("step number: " + str(curr_step), fontsize=46)
        plt.savefig(img_end_file, format='PNG')

        gui.Popup('Log files saved in ' + str(folder_path))
    

    if not started and event == 'Generate random graph':
        graph_types = [
            'Dense',
            'Erdős-Rényi',
            'Powerlaw Cluster',
            'Small World'
        ]
        generate_layout = [
            [gui.Text('Graph type: '), gui.Combo(graph_types, default_value=graph_types[0], readonly=True, key='-GRAPHTYPE-')],
            [gui.Text('Nodes: '), gui.InputText('10', size=6, key='-N-')],
            [gui.Text('Edges (Dense, Powerlaw): '), gui.InputText('15', size=6, key='-M-')],
            [gui.Text('Probability: '), gui.InputText(str(random.random()), size=22, key='-P-'), gui.Button('Random p')],
            [gui.Text('K Nearest Neighbors (Small World): '), gui.InputText('5', size=6, key='-K-')],
            [gui.Text('Seed: '), gui.InputText(str(random.randint(10000, 9999999)), size=8, key='-SEED-'), gui.Button('Random seed')],
            [gui.Button('Generate'), gui.Button('Cancel')]
        ]

        generate_window = gui.Window('Generate Random Graph', layout=generate_layout, modal=True, size=(450, 300))

        while True:
            event_gen, values_gen = generate_window.read()
            print(event_gen, values_gen)

            if event_gen == 'Random p':
                generate_window['-P-'].update(value=str(random.random()))
            
            if event_gen == 'Random seed':
                generate_window['-SEED-'].update(value=str(random.randint(10000, 9999999)))

            if event_gen == 'Generate':

                seed = int(values_gen['-SEED-']) if values_gen['-SEED-'] != '' else random.randint()
                n = int(values_gen['-N-']) if int(values_gen['-N-']) > 1 else 1
                g_type = values_gen['-GRAPHTYPE-']

                if g_type == 'Dense':
                    m = int(values_gen['-M-']) if int(values_gen['-M-']) > 1 else 1
                    g = nx.dense_gnm_random_graph(n, m, seed)
                    graph_origin = "Random Dense Graph"
                    graph_origin_info = "\tSeed: " + str(seed)
                elif g_type == 'Erdős-Rényi':
                    p = float(values_gen['-P-']) if float(values_gen['-P-']) >= 0.0 and float(values_gen['-P-']) <= 1.0 else 0.5
                    g = nx.fast_gnp_random_graph(n, p, seed, directed=False)
                    graph_origin = "Random Erdos-Renyi Graph"
                    graph_origin_info = "\tSeed: " + str(seed) + "\n" + "\tP: " + str(p)
                elif g_type == 'Powerlaw Cluster':
                    m = int(values_gen['-M-']) if int(values_gen['-M-']) >= 1 and int(values_gen['-M-']) <= n else 1
                    p = float(values_gen['-P-']) if float(values_gen['-P-']) >= 0.0 and float(values_gen['-P-']) <= 1.0 else 0.5
                    g = nx.powerlaw_cluster_graph(n, m, p, seed)
                    graph_origin = "Random Powerlaw Graph"
                    graph_origin_info = "\tSeed: " + str(seed) + "\n" + "\tP: " + str(p) + "\n" + "\tEdges for each new node (m): " + str(m)
                elif g_type == 'Small World':
                    k = int(values_gen['-K-']) if int(values_gen['-K-']) > 0 and int(values_gen['-K-']) <= n else int(n/2)
                    p = float(values_gen['-P-']) if float(values_gen['-P-']) >= 0.0 and float(values_gen['-P-']) <= 1.0 else 0.5
                    tries = 100
                    g = nx.connected_watts_strogatz_graph(n, k, p, tries, seed)
                    graph_origin = "Random Small World Graph"
                    graph_origin_info = "\tSeed: " + str(seed) + "\n" + "\tP: " + str(p) + "\n" + "\tK-Nearest neighbors to join: " + str(k)
                
                node_opinions = randomize_node_opinions(g, opinions)
                
                (tk_image, size) = get_graph_tk_image(g)
                window['-MAIN IMAGE-'].update(data=tk_image, size=size)

                break


            if event_gen == gui.WIN_CLOSED or event_gen == 'Cancel':
                break
        
        generate_window.close()
    

    if event == 'Save Graph as csv':
        if g is None:
            gui.popup('There\'s no graph to save')
            continue
        
        filename = gui.tk.filedialog.asksaveasfilename(
            filetypes=[('csv', '.csv')],
            defaultextension=[('csv', '.csv')],
        )

        if filename == '':
            continue

        adj = nx.to_pandas_adjacency(g)

        ops = []
        res = []

        for node in g:
            op = node_opinions[node]
            ops.append(op)
            res.append(opinions[op])
        
        adj['opinions'] = ops
        adj['resistence'] = res

        adj.to_csv(filename, header=False, index=False)



    if not started and event == 'Load Graph':
        filename = gui.popup_get_file('load graph from csv file', no_window=True, file_types=[('csv', '.csv')])
        if filename == '':
            continue
        
        try:
            (g, opinions, node_opinions) = load_csv_graph(filename)
            graph_origin = "Loaded from csv file:"
            graph_origin_info = str(filename)
        except RuntimeError:
            gui.popup('csv file not formatted correctly')
            continue

        # Draw the graph
        (tk_image, size) = get_graph_tk_image(g)
        window['-MAIN IMAGE-'].update(data=tk_image, size=size)

    if not started and event == 'Nodes':
        if g is None:
            continue

        node_edit_layout = [
            [gui.Text('Node'), gui.Text('Opinion'), gui.Button('All Random'), gui.HorizontalSeparator(), gui.Button('Save'), gui.Button('Cancel')],
            [gui.Column(layout=[
                [gui.Column(layout=[
                    [
                        gui.Text(str(i), key='key'+str(i)),
                        gui.Combo(list(opinions.keys()), default_value=node_opinions[i], key='val'+str(i), readonly=True) if i in list(node_opinions.keys()) else gui.Combo(list(opinions.keys()), default_value=list(opinions.keys())[0], key='val'+str(i), readonly=True),
                        gui.Button(button_text='random', key='rand'+str(i))
                    ]], key=i)
                ] for i in list(g.nodes)
                ], key='-COLUMN-', scrollable=True, vertical_scroll_only=True, expand_x=True, expand_y=True)]
        ]

        node_edit_window = gui.Window('Edit Nodes', layout=node_edit_layout, modal=True, resizable=True, size=(450, 600))

        while True:
            event_nd, values_nd = node_edit_window.read()
            print(event_nd, values_nd)

            if isinstance(event_nd, str) and event_nd.startswith('rand'):
                index = int(event_nd[4:])
                r = random.randint(0, len(opinions)-1)
                rand_opinion = list(opinions.keys())[r]
                node_edit_window['val'+str(index)].update(rand_opinion)
            
            if event_nd == 'All Random':
                for i in list(g.nodes):
                    r = random.randint(0, len(opinions)-1)
                    rand_opinion = list(opinions.keys())[r]
                    node_edit_window['val'+str(i)].update(rand_opinion)
            
            if event_nd == 'Save':
                node_opinions = {}
                for i in list(g.nodes):
                    node_opinions[i] = values_nd['val'+str(i)]
                print(node_opinions)
                break

            if event_nd == gui.WIN_CLOSED or event_nd == 'Cancel':
                break
        
        node_edit_window.close()

    if not started and event == 'Opinions':
        curr_opnum = len(opinions)
        opinion_edit_layout = [
            [gui.InputText('1', size=5, key='-NEWOP-'), gui.Button('Add Opinions'), gui.Button('All Random'), gui.Text(text='opinions: '+str(curr_opnum), key='-OPNUM-'), gui.HorizontalSeparator(), gui.Button('Save'), gui.Button('Cancel')],
            [gui.Column(layout=[
                [gui.Column(layout=[
                    [
                        gui.InputText(default_text=o, size=10, key='key'+str(i)),
                        gui.InputText(default_text=opinions[o], size=20, key='val'+str(i)),
                        gui.Button(button_text='random', key='rand'+str(i)),
                        gui.Button(button_text='delete', key='del'+str(i))
                    ]], key=i)
                ] for (i, o) in enumerate(opinions)
                ], key='-COLUMN-', scrollable=True, vertical_scroll_only=True, expand_x=True, expand_y=True)]
        ]        
        opinion_edit_window = gui.Window('Edit Opinions', layout=opinion_edit_layout, modal=True, resizable=True, size=(450, 600))
        i = len(opinions)

        while True:
            event_op, values_op = opinion_edit_window.read()
            print(event_op, values_op)

            if event_op == 'Add Opinions':
                to_add = int(values_op['-NEWOP-'])
                if to_add < 1:
                    to_add = 1
                for _ in range(to_add):
                    opinion_edit_window.extend_layout(opinion_edit_window['-COLUMN-'], [[
                        gui.Column(layout=[[
                            gui.InputText(default_text=random_str(opinions.keys()), size=10, key='key'+str(i)),
                            gui.InputText(default_text=str(random.random()), size=20, key='val'+str(i)),
                            gui.Button(button_text='random', key='rand'+str(i)),
                            gui.Button(button_text='delete', key='del'+str(i))
                        ]], key=i)
                    ]])
                    i += 1
                    curr_opnum += 1
                    opinion_edit_window['-OPNUM-'].update('opinions: '+str(curr_opnum))
            
            if event_op == 'All Random':
                for j in range(i):
                    opinion_edit_window['val'+str(j)].update(str(random.random()))
                    if values_op['key'+str(j)] == '':
                        opinion_edit_window['key'+str(j)].update(random_str(opinions.keys()))

            
            if isinstance(event_op, str) and event_op.startswith('rand'):
                index = int(event_op[4:])
                opinion_edit_window['val'+str(index)].update(str(random.random()))
                if values_op['key'+str(index)] == '':
                    opinion_edit_window['key'+str(index)].update(random_str(opinions.keys()))
            
            if isinstance(event_op, str) and event_op.startswith('del'):
                index = int(event_op[3:])
                opinion_edit_window[index].update(visible=False)
                opinion_edit_window[index].hide_row()
                curr_opnum -= 1
                opinion_edit_window['-OPNUM-'].update('opinions: '+str(curr_opnum))



            if event_op == 'Save':
                opinions = {}
                c = False
                for j in range(i):
                    o = values_op['key'+str(j)]
                    v = values_op['val'+str(j)]
                    if opinion_edit_window[j].visible:
                        if v == '':
                            v = 0
                        try:
                            v = float(v)
                            if v > 1:
                                raise ValueError
                        except ValueError:
                            v = random.random()
                            c = True
                        opinions[o] = v
                print(opinions)
                if c:
                    gui.popup_ok("WARNING: some inserted values aren't correct (they must be numbers between 0 and 1), random values used instead")
                break

            if event_op == gui.WIN_CLOSED or event_op == 'Cancel':
                break

        opinion_edit_window.close()
    
    if event == gui.WIN_CLOSED:
        break

window.close()