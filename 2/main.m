pkg load interval
%addpath(genpath('../octave-interval-examples/m'))
%addpath(genpath('./m'))

load("data.mat")
x = data(:, 1);
y = data(:, 2);

%draw_graph(x, y, "n", "mV", "", 1, "all_data.eps");
%print -djpg all_data.jpg
%draw_graph(x, y, "time", "value", "", 1, "filtered_data.eps");
%draw_graph(x, y, "time", "value", "d", 1, "selected_data.eps");

%dot_problem(x, y);

irp_temp = interval_problem(x, y);
[b_maxdiag, b_gravity] = parameters(x, y, irp_temp);
joint_depth(irp_temp, b_maxdiag, b_gravity);
%prediction(x, y, irp_temp, b_maxdiag, b_gravity);
%edje_points(x, y, irp_temp);
