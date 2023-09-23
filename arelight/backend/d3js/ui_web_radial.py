ui_web_radial_template = """
    <!DOCTYPE html>
    <style>
        .node {
            font: 10px sans-serif;
        }

        .link {
            stroke: steelblue;
            stroke-opacity: 0.5;
            fill: none;
            pointer-events: none;
        }
    </style>
    <body>
    <script src="https://d3js.org/d3.v4.min.js"></script>
    <script>
        var transition_time = 200
        var diameter = 960,
            radius = diameter / 2,
            innerRadius = radius - 120;
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
        var svg = d3.select("body").append("svg")
            .attr("id", "svg")
            .attr("width", diameter)
            .attr("height", diameter)
            .append("g")
            .attr("transform", "translate(" + radius + "," + radius + ")");
        var link = svg.append("g").selectAll(".link"),
            node = svg.append("g").selectAll(".node");

        d3.json("<SOURCE_JSON_FILEPATH>", function (error, classes) {
            if (error) throw error;
            function calculate_opacity(d) {
                var target = d.target.data.name, weight = undefined, sent = undefined
                for (const source of d.source.data.imports) {
                    if (source.name === target) {weight = source.w, sent = source.sent}
                }
                if (weight !== undefined) {return weight * 10}
                return 1
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
                    if (sent === "pos") {return "green"}
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
                    if (weight !== undefined) {return weight * 10}
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
                }).on('mouseover', function (d_parent) {
                    name = d_parent.data.name
                    d3.selectAll("[id*='***']")
                        .transition().duration(transition_time)
                        .attr('opacity', 0)
                        //.style('stroke', 'grey');
                    d3.selectAll("[id*='" + name + "']")
                        .transition().duration(transition_time)
                        .attr('opacity', function (d) {return calculate_opacity(d)})
                        //.style('stroke', function (d) {return calculate_color(d)});
                }).on('mouseout', function (d_parent) {
                    name = d_parent.data.name
                    d3.selectAll("[id*='***']")
                        .transition().duration(transition_time)
                        .attr('opacity', function (d) {return calculate_opacity(d)})
                        //.style('stroke', function (d) {return calculate_color(d)});
                })
        });

        // Lazily construct the package hierarchy from class names.
        function packageHierarchy(classes) {
            console.log(classes)
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
    </script>
"""


def get_radial_web_ui(json_data_at_server_filepath):
    """ json_data_at_server_filepath: str
            source to the data for visualization
    """
    html_content = ui_web_radial_template.replace(
        "<SOURCE_JSON_FILEPATH>", json_data_at_server_filepath)

    return html_content
