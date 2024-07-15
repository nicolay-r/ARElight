import json
import os
from os.path import join

from arekit.common.utils import create_dir_if_not_exists

ui_template = """
<!DOCTYPE html>
<meta charset="utf-8">
    <style>
        .links line {
          stroke: #999;
          stroke-opacity: 0.6;
        }
        .nodes circle {
          stroke: #fff;
          stroke-width: 1.5px;
        }

        .link {
            stroke: steelblue;
            stroke-opacity: 0.5;
            fill: none;
            pointer-events: none;
        }
        text {
          font: 10px sans-serif;
          /*pointer-events: none;*/
          text-shadow: 0 1px 0 #fff, 1px 0 0 #fff, 0 -1px 0 #fff, -1px 0 0 #fff;
        }
        .red-text {
            color: #b42323;
        }
        .blue-text {
            color: #2850b4
        }
        .grey-text {
            color: #979797;
        }
    </style>

    <body>
        <head>
          <title>ARElight-0.25.0 Demo [LEGACY]</title>
          <meta charset="utf-8">
          <meta name="viewport" content="width=device-width, initial-scale=1">
          <link rel="stylesheet" href="https://maxcdn.bootstrapcdn.com/bootstrap/3.4.1/css/bootstrap.min.css">
          <script src="https://ajax.googleapis.com/ajax/libs/jquery/3.6.4/jquery.min.js"></script>
          <script src="https://maxcdn.bootstrapcdn.com/bootstrap/3.4.1/js/bootstrap.min.js"></script>
        </head>

        <div class="container">
          <h1>ARElight-0.25.0 Demo [LEGACY]</h1>
        </div>

        <div class="container">
            <div class="row">
                <div class="col-sm-4">
                    <div>
                        <div id="dataset_folder">
                            Here is your datasets from folder <!--INSERT_FOLDER_NAME-->
                        </div>
                        <select id="datasets_selector" class="form-select" aria-label="Default select example" onchange="selected_new_dataset()">
                            <!--INSERT_DATASETS_NAMES-->
                        </select>
                        <div style="padding-top: 10px">
                            <p id="dataset_description"></p>
                        </div>
                        <div style="padding-top: 10px">
                            <div class="form-group">
                                <label for="node_freq">Vertex frequency:</label>
                                <input type="number" class="form-control" id="node_freq" placeholder="100" value="100" onchange="selected_new_dataset()">
                                <small class="form-text text-muted">Leave top x% of Vertices by their frequency in text.</small>
                            </div>
                            <div class="form-group">
                                <label for="edge_scale">Edge width:</label>
                                <input type="number" class="form-control" id="edge_scale" placeholder="100" value="100" onchange="selected_new_dataset()">
                                <small class="form-text text-muted">Scale x% of thickness for edges if you need.</small>
                            </div>
                            <div class="form-group">
                                <label for="edge_opacity">Edge opacity:</label>
                                <input type="number" class="form-control" id="edge_opacity" placeholder="0.5" value="0.5" onchange="selected_new_dataset()">
                                <small class="form-text text-muted">Scale x% of opacity for edges if you need from 0 to 1.</small>
                            </div>
                            <div class="form-group">
                                <label for="force_scale">Force scale:</label>
                                <input type="number" class="form-control" id="force_scale" placeholder="100" value="100" onchange="selected_new_dataset()">
                                <small class="form-text text-muted">Only for force graph: vertex repulsion force.</small>
                            </div>
                            <div class="form-check">
                                <input class="form-check-input" type="checkbox" value="" id="check_positive" checked onchange="selected_new_dataset()">
                                <label class="form-check-label" for="check_positive">Display <span class="blue-text">positive</span> edges</label>
                            </div>
                            <div class="form-check">
                                <input class="form-check-input" type="checkbox" value="" id="check_negative" checked onchange="selected_new_dataset()">
                                <label class="form-check-label" for="check_negative">Display <span class="red-text">negative</span> edges</label>
                            </div>
                            <div class="form-check">
                                <input class="form-check-input" type="checkbox" value="" id="check_neutral" checked onchange="selected_new_dataset()">
                                <label class="form-check-label" for="check_neutral">Display <span class="grey-text">neutral</span> edges</label>
                            </div>
                            <small class="form-text text-muted">Remove unnecessary links of you need.</small>

                        </div>
                    </div>

                </div>

                <div class="col-sm-8">

                    <div class="row-xs-1 center-block" >
                        <button id="force_BTN" type="button" class="btn btn-primary" onclick="force_BTN()">
                            Force Layout
                        </button>
                        <button id="radial_BTN" type="button" class="btn btn-primary" onclick="radial_BTN()">
                            Radial Layout
                        </button>
                    </div>
                    <div style="padding-top: 10px" align="center" id="svg_container">
                        <svg></svg>
                    </div>
                </div>
            </div>
        </div>

    <script src="https://d3js.org/d3.v4.min.js"></script>
    <script>

        folder_with_datasets_path = "HOST_ROOT_PATH"

        height = window.innerHeight*0.8;
        width = window.innerWidth*0.5;
        if (height < 100){
            height = width;
        }

        selected_vis = "force"

        document.querySelector('#force_BTN').classList.add('disabled');
        document.querySelector('#radial_BTN').classList.add('disabled');

        function selected_new_dataset() {
            var selectobject = document.getElementById("datasets_selector");
            for (var i=0; i<selectobject.length; i++) {
                if (selectobject.options[i].value == 'select')
                    selectobject.remove(i);
            }
            if (selected_vis === "force"){
                force_BTN()
            }
            if (selected_vis === "radial"){
                radial_BTN()
            }
            dataset_name = d3.select("#datasets_selector").node().value;
            draw_description(dataset_name)
        }

        function force_BTN() {
            selected_vis = "force"
            document.querySelector('#radial_BTN').classList.remove('disabled');
            document.querySelector('#force_BTN').classList.remove('disabled');
            document.querySelector('#force_BTN').classList.add('disabled');
            dataset_name = d3.select("#datasets_selector").node().value;
            draw_force(dataset_name)
        }

        function radial_BTN() {
            selected_vis = "radial"
            document.querySelector('#radial_BTN').classList.remove('disabled');
            document.querySelector('#force_BTN').classList.remove('disabled');
            document.querySelector('#radial_BTN').classList.add('disabled');
            dataset_name = d3.select("#datasets_selector").node().value;
            draw_radial(dataset_name)
        }

        function draw_description(dataset_name) {
            d3.json(folder_with_datasets_path+"descriptions/"+dataset_name, function(data) {
                d3.select("#dataset_description").text(data.description)
            })
        }

        function get_draw_parameters() {
            var links_scale = d3.select("#edge_scale").property("value")
            if (isNaN(links_scale)){links_scale = 100}
            var check_positive = d3.select("#check_positive").property("checked");
            var check_negative = d3.select("#check_negative").property("checked");
            var check_neutral = d3.select("#check_neutral").property("checked");
            var opacity_scale = d3.select("#edge_opacity").property("value")
            if (isNaN(opacity_scale)){opacity_scale = 0.5}
            var force_scale = d3.select("#force_scale").property("value")
            if (isNaN(force_scale)){force_scale = 0.5}
            var node_freq = d3.select("#node_freq").property("value")
            if (isNaN(node_freq)){node_freq = 100}
            return [links_scale/100, check_positive, check_negative, check_neutral, opacity_scale, force_scale, node_freq/100]
        }


        //filter nodes
        function filterNodes(array, percent) {
            // Sort the array in descending order based on the 'c' property.
            array.sort((a, b) => b.c - a.c);
            // Calculate the number of items to keep.
            const itemsToKeep = Math.ceil(array.length * (percent));
            // Return the top x% of items.
            return array.slice(0, itemsToKeep);
        }
        //filter links
        function filterLinksForce(nodes, links) {
            // Create a set of all ids in the nodes array for efficient look-up.
            const nodeIds = new Set(nodes.map(node => node.id));
            // Filter the links array to keep only those objects where both the source
            // and target properties appear in the nodeIds set.
            return links.filter(link => nodeIds.has(link.source) && nodeIds.has(link.target));
        }
        function filterLinksRadial(nodes) {
            // Create a set of all ids in the nodes array for efficient look-up.
            const nodeIds = new Set(nodes.map(node => node.name));
            // Filter the links array to keep only those objects where both the source
            // and target properties appear in the nodeIds set.
            for (var i=0; i<nodes.length; i++){
                nodes[i].imports = nodes[i].imports.filter(link => nodeIds.has(link.name))
            }
            return nodes;
        }


        function draw_force(dataset_name){
            d3.select("svg").remove()
            var draw_parameters = get_draw_parameters();

            d3.json(folder_with_datasets_path+"force/"+dataset_name, function(error, graph) {


                graph.nodes = filterNodes(graph.nodes, draw_parameters[6])
                graph.links = filterLinksForce(graph.nodes, graph.links)


                var svg = d3.select("#svg_container").append("svg")
                svg.attr("width", width);
                svg.attr("height", height);
                svg.attr("style", "outline: thin solid black;");

                var color = d3.scaleLinear().range(['blue', 'red']);

                var simulation = d3.forceSimulation()
                 .force("link", d3.forceLink().id(function(d) { return d.id; }).distance(150))
                 .force("charge", d3.forceManyBody().strength(-1*draw_parameters[5]))
                 .force("center", d3.forceCenter(width / 2, height / 2));

                var link = svg.append("g")
                  .attr("class", "links")
                  .selectAll("line")
                 .data(graph.links)
                 .enter().append("line")
                  //.attr("stroke-width", function(d) { return Math.log(d.c)/ Math.LN10 * 5; })
                  //.attr('opacity', function(d) { o = Math.log(d.c/100); if (o<0.1){return 0.1}; return o; })
                  .attr("stroke-width", function(d) { return d.c*30*draw_parameters[0]})
                  //.attr("stroke-width", function(d) { return d.c/10 })
                  .attr('opacity', function(d)  {
                      if (d.sent === "pos" && !draw_parameters[1]){return 0;}
                      if (d.sent === "neg" && !draw_parameters[2]){return 0;}
                      if (d.sent === "neu" && !draw_parameters[3]){return 0;}
                      return draw_parameters[4];
                  })
                  .style('stroke', function(d) {
                      if (d.sent === "pos"){return "blue";}
                      if (d.sent === "neg"){return "red";}
                      if (d.sent === "neu"){return "grey";}
                      return "pink";
                  })
                  //.style('stroke', function(d) { return "red"; })


                var node = svg.selectAll(".node")
                            .data(graph.nodes)
                            .enter().append("g")
                                        .attr("class", "node")
                             .call(d3.drag()
                      .on("start", dragstarted)
                      .on("drag", dragged)
                      .on("end", dragended));

                node.append("circle")
                  .attr("r", function(d) { return 10; })
                  //.attr("r", function(d) { return Math.log(d.c) / Math.LN10 * 5; })
                  //.attr("fill", function(d) { return color(d.m_m); });
                  .attr("fill", function(d) { return "black"; })
                  .attr('fill-opacity', function(d) { return d.c / 17249; })
                  ;

                node.append("text")
                // 	.attr("dx", 6)
                    .text(function(d) { return d.id; });

                //    link.append("text")
                // 	.attr("dx", 6)
                //    .text(function(d) { return d.value; });

                 simulation
                   .nodes(graph.nodes)
                   .on("tick", ticked);

                 simulation.force("link")
                   .links(graph.links);

                 function ticked() {
                   link
                    .attr("x1", function(d) { return d.source.x; })
                    .attr("y1", function(d) { return d.source.y; })
                    .attr("x2", function(d) { return d.target.x; })
                    .attr("y2", function(d) { return d.target.y; });

                node.attr("transform", function(d) {
                                padding = 10
                                if (d.x>width-padding){d.x=width-padding}
                                if (d.x<padding){d.x=padding}
                                if (d.y>height-padding){d.y=height-padding}
                                if (d.y<padding){d.y=padding}
                                return "translate(" + d.x + "," + d.y + ")";
                            });
                   }

                 function dragstarted(d) {
                  if (!d3.event.active) simulation.alphaTarget(0.3).restart();
                  d.fx = d.x;
                  d.fy = d.y;
                  }

                 function dragged(d) {
                   d.fx = d3.event.x;
                   d.fy = d3.event.y;
                 }

                 function dragended(d) {
                    if (!d3.event.active) simulation.alphaTarget(0);
                    d.fx = null;
                    d.fy = null;
                 }
        });
        }

        function draw_radial(dataset_name) {
            d3.select("svg").remove()
            var draw_parameters = get_draw_parameters();

            var transition_time = 200
            var diameter = width,
                radius = diameter / 2,
                innerRadius = radius * 0.8;
            var cluster = d3.cluster()
                .size([360, innerRadius]);
            var line = d3.radialLine()
                .curve(d3.curveBundle.beta(0.85))
                .radius(function (d) {
                    return d.y;
                })
                .angle(function (d) {
                    return d.x / 180 * Math.PI;
                });

            var svg = d3.select("#svg_container").append("svg");
            svg.attr("style", "outline: thin solid black;");
            svg
                .attr("id", "svg")
                .attr("width", diameter)
                .attr("height", diameter);
            svg = svg
                .append("g")
                .attr("transform", "translate(" + radius + "," + radius + ")");
            var link = svg.append("g").selectAll(".link"),
                node = svg.append("g").selectAll(".node");

            d3.json(folder_with_datasets_path+"radial/"+dataset_name, function (error, classes) {
                if (error) throw error;

                console.log(classes)
                classes = filterNodes(classes, draw_parameters[6])
                classes = filterLinksRadial(classes)
                console.log(classes)


                function calculate_opacity(d) {
                    var target = d.target.data.name, weight = undefined, sent = undefined
                    for (const source of d.source.data.imports) {
                        if (source.name === target) {weight = source.w, sent = source.sent}
                    }
                    if (sent === "pos" && !draw_parameters[1]){return 0;}
                    if (sent === "neg" && !draw_parameters[2]){return 0;}
                    if (sent === "neu" && !draw_parameters[3]){return 0;}

                    if (weight !== undefined) {return weight * 10 * draw_parameters[4]}

                    return 1 * draw_parameters[4]
                }

                function calculate_color(d){
                    var target = d.target.data.name, weight = undefined, sent = undefined
                        for (const source of d.source.data.imports) {
                            if (source.name === target) {
                                weight = source.w
                                sent = source.sent
                            }
                        }
                        if (sent === "neg") {return "red"}
                        if (sent === "pos") {return "blue"}
                        if (sent === "neu") {return "grey"}
                        return "black"
                }

                var root = packageHierarchy(classes).sum(function (d) {return d.w;});
                cluster(root);
                link = link
                    .data(packageImports(root.leaves()))
                    .enter().append("path")
                    .each(function (d) {d.source = d[0], d.target = d[d.length - 1];})
                    .attr("class", "link")
                    .attr("d", line)
                    .attr("id", function (d) {
                        if (d.data !== undefined) {return null}
                        return d.target.data.name + "***" + d.source.data.name
                    })
                    .attr("stroke-width", function (d) {
                        var target = d.target.data.name, weight = undefined, sent = undefined
                        for (const source of d.source.data.imports) {
                            if (source.name === target) {weight = source.w, sent = source.sent}
                        }
                        if (weight !== undefined) {return weight * 10 * draw_parameters[0]}
                        return 1
                    })
                    .attr('opacity', function (d) {return calculate_opacity(d)})
                    .style("stroke", function (d) {return calculate_color(d)})


                node = node
                    .data(root.leaves())
                    .enter().append("text")
                    .attr("class", "node")
                    .attr("dy", "0.31em")
                    .attr("transform", function (d) {return "rotate(" + (d.x - 90) + ")translate(" + (d.y + 8) + ",0)" + (d.x < 180 ? "" : "rotate(180)");})
                    .attr("text-anchor", function (d) {return d.x < 180 ? "start" : "end";})
                    .text(function (d) {
                        return d.data.key;
                    })
                    .on('mouseover', function (d_parent) {
                        name = d_parent.data.name
                        d3.selectAll("[id*='***']")
                            .transition().duration(transition_time)
                            .attr('opacity', 0)
                            //.style('stroke', 'grey');
                        d3.selectAll("[id*='" + name + "']")
                            .transition().duration(transition_time)
                            .attr('opacity', function (d) {return calculate_opacity(d)})
                            //.style('stroke', function (d) {return calculate_color(d)});
                    })
                    .on('mouseout', function (d_parent) {
                        name = d_parent.data.name
                        d3.selectAll("[id*='***']")
                            .transition().duration(transition_time)
                            .attr('opacity', function (d) {return calculate_opacity(d)})
                            //.style('stroke', function (d) {return calculate_color(d)});
                    })
            });

            // Lazily construct the package hierarchy from class names.
            function packageHierarchy(classes) {
                var map = {};

                function find(name, data) {
                    var node = map[name], i;
                    if (!node) {
                        node = map[name] = data || {name: name, children: []};
                        if (name.length) {
                            node.parent = find(name.substring(0, i = name.lastIndexOf(".")));
                            node.parent.children.push(node);
                            node.key = name.substring(i + 1);
                        }
                    }
                    return node;
                }

                classes.forEach(function (d) {
                    find(d.name, d);
                });
                return d3.hierarchy(map[""]);
            }

            // Return a list of imports for the given array of nodes.
            function packageImports(nodes) {
                var map = {},
                    imports = [];
                // Compute a map from name to node.
                nodes.forEach(function (d) {
                    map[d.data.name] = d;
                });
                // For each import, construct a link from the source to target node.
                nodes.forEach(function (d) {
                    if (d.data.imports) d.data.imports.forEach(function (i) {
                        //console.log(d)
                        // d.data.w_l = i["w"]
                        // d.data.sent = i["sent"]
                        imports.push(map[d.data.name].path(map[i["name"]]));
                    });
                });
                return imports;
            }
            }
    </script>
    </body>
</html>
"""


def get_web_ui(datasets_list, host_root_path, folder_name=""):
    """ datasets_list: list
            list of processed datasets that stored in output folder
        folder_name: str
            name of folder with datasets
    """

    dataset_options = []
    for dataset in datasets_list:
        dataset_options.append(f'<option value="{dataset}.json">{dataset}</option>\n')

    html_content = ui_template\
        .replace("<!--INSERT_DATASETS_NAMES-->", "\n".join(dataset_options))\
        .replace("<!--INSERT_FOLDER_NAME-->", folder_name)\
        .replace("HOST_ROOT_PATH", host_root_path)

    return html_content


GRAPH_TYPE_RADIAL = 'radial'
GRAPH_TYPE_FORCE = 'force'


def iter_ui_backend_folders(keep_graph=False, keep_desc=False):
    if keep_graph:
       yield GRAPH_TYPE_RADIAL
       yield GRAPH_TYPE_FORCE
    if keep_desc:
       yield "descriptions"


def save_demo_page(target_dir, host_root_path, collection_name=None, desc_name=None, desc_labels=None):

    descriptions_dir = join(target_dir, next(iter_ui_backend_folders(keep_desc=True)))
    create_dir_if_not_exists(filepath=join(descriptions_dir, "__placeholder__"))

    # Add new collection and expand with existed.
    suffix = '.json'
    descriptors = [filename[:-len(suffix)] for filename in os.listdir(descriptions_dir)
                   if filename.endswith(suffix)]

    # Save Graph description.
    if collection_name is not None:
        desc_path = join(descriptions_dir, f"{collection_name}{suffix}")
        with open(desc_path, "w") as f:
            f.write(json.dumps({
                "description": desc_name if descriptors is not None else collection_name,
                "labels": desc_labels
            }))

        # Place collection name on to of the list.
        if collection_name in descriptors:
            del descriptors[descriptors.index(collection_name)]
        descriptors = [collection_name] + descriptors

    # Demo content.
    html_content = get_web_ui(datasets_list=descriptors, host_root_path=host_root_path)
    with open(join(target_dir, "index.html"), "w") as f_out:
        f_out.write(html_content)
