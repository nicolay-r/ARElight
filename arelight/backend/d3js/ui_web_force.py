import json
from os import makedirs
from os.path import join, exists

ui_web_force_template = """
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


    text {
      font: 10px sans-serif;
      pointer-events: none;
      text-shadow: 0 1px 0 #fff, 1px 0 0 #fff, 0 -1px 0 #fff, -1px 0 0 #fff;
    }

    </style>
    <svg width="1960" height="1600"></svg>
    <script src="https://d3js.org/d3.v4.min.js"></script>
    <script>

    d3.json("<SOURCE_JSON_FILEPATH>", function(error, graph) {
      var svg = d3.select("svg"),
        width = +svg.attr("width"),
        height = +svg.attr("height");

    var color = d3.scaleLinear().range(['blue', 'red']);;

    var simulation = d3.forceSimulation()
     .force("link", d3.forceLink().id(function(d) { return d.id; }).distance(150))
     .force("charge", d3.forceManyBody().strength(-300))
     .force("center", d3.forceCenter(width / 2, height / 2));

    var link = svg.append("g")
      .attr("class", "links")
      .selectAll("line")
     .data(graph.links)
     .enter().append("line")
      //.attr("stroke-width", function(d) { return Math.log(d.c)/ Math.LN10 * 5; })
      //.attr('opacity', function(d) { o = Math.log(d.c/100); if (o<0.1){return 0.1}; return o; })
      .attr("stroke-width", function(d) { return d.c*30 })
      //.attr("stroke-width", function(d) { return d.c/10 })
      .attr('opacity', function(d)  {return 0.5; })
      .style('stroke', function(d) { if (d.sent == "neg"){return "red"}; return "blue"; })
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
    </script>
"""


def get_force_web_ui(json_data_at_server_filepath):
    """ json_data_at_server_filepath: str
            source to the data for visualization
    """
    html_content = ui_web_force_template.replace(
        "<SOURCE_JSON_FILEPATH>", json_data_at_server_filepath)

    return html_content

