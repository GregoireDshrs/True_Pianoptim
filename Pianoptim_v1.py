"""
 !! Les axes du modèle ne sont pas les mêmes que ceux généralement utilisés en biomécanique : x axe de flexion, y supination/pronation, z vertical
 ici on a : Y -» X , Z-» Y et X -» Z
 """
import biorbd_casadi as biorbd
from bioptim import (
    PenaltyNode,
    OptimalControlProgram,
    DynamicsList,
    DynamicsFcn,
    ObjectiveList,
    ObjectiveFcn,
    ConstraintList,
    ConstraintFcn,
    BoundsList,
    QAndQDotBounds,
    InitialGuessList,
    OdeSolver,
    Node,
    Axis,
    Solver,
    PhaseTransitionList,
    PhaseTransitionFcn,
    PlotType,

)


# Constants from data collect
# velocities of the right hand of subject 134
vel_x_0 = 0.52489776
vel_x_1 = -0.00321311
vel_x_2 = 0.64569518
vel_x_3 = 0.04628122
vel_x_4 = -0.01133422
vel_x_5 = 0.31271958

vel_y_0 = 0.54408214
vel_y_1 = 0.01283999
vel_y_2 = 0.28540601
vel_y_3 = 0.02580192
vel_y_4 = 0.09021791
vel_y_5 = 0.42298668

vel_z_0 = 0.7477114
vel_z_1 = 0.17580993
vel_z_2 = 0.6360936
vel_z_3 = 0.3468823
vel_z_4 = -0.03609537
vel_z_5 = 0.38915613

pos_x_0 = 0.10665989
pos_x_1 = 0.25553592
pos_x_2 = 0.26675006
pos_x_3 = 0.39711206
pos_x_4 = 0.38712035
pos_x_5 = 0.9867809

pos_y_0 = 0.40344472
pos_y_1 = 0.37350889
pos_y_2 = 0.38965598
pos_y_3 = 0.34155492
pos_y_4 = 0.33750396
pos_y_5 = 0.39703432

pos_z_0 = 2.76551207
pos_z_1 = 2.74265983
pos_z_2 = 2.76575107
pos_z_3 = 2.73511557
pos_z_4 = 2.72985087
pos_z_5 = 2.75654283


stdev_vel_x_0 = 0.12266391
stdev_vel_x_1 = 0.05459328
stdev_vel_x_2 = 0.08348852
stdev_vel_x_3 = 0.06236412
stdev_vel_x_4 = 0.06251115
stdev_vel_x_5 = 0.10486219

# stdev_vel_y_0 = 0.06590577
# stdev_vel_y_1 = 0.04433499
# stdev_vel_y_2 = 0.08251966
# stdev_vel_y_3 = 0.03813032
# stdev_vel_y_4 = 0.07607116
# stdev_vel_y_5 = 0.0713205

stdev_vel_z_0 = 0.11591871
stdev_vel_z_1 = 0.10771169
stdev_vel_z_2 = 0.081717
stdev_vel_z_3 = 0.09894744
stdev_vel_z_4 = 0.11820802
stdev_vel_z_5 = 0.1479469



mean_time_phase_0 = 0.36574653   # phase 0 is from first marker to second marker
mean_time_phase_1 = 0.10555556
mean_time_phase_2 = 0.40625
mean_time_phase_3 = 0.10387153
mean_time_phase_4 = 1.00338542  # phase 4 is from last marker to first marker

def minimize_difference(all_pn: PenaltyNode):
    return all_pn[0].nlp.controls.cx_end - all_pn[1].nlp.controls.cx

#def custom_func_track_markers():

    #return markers_diff


def prepare_ocp(
    biorbd_model_path: str = "Piano_final_version.bioMod",
        ode_solver: OdeSolver = OdeSolver.RK4(),
        long_optim: bool = False,
) -> OptimalControlProgram:
    """
    Prepare the ocp

    Parameters
    ----------
    biorbd_model_path: str
        The path to the bioMod
    ode_solver: OdeSolver
        The ode solve to use
    long_optim: bool
        If the solver should solve the precise optimization (500 shooting points) or the approximate (50 points)

    Returns
    -------
    The OptimalControlProgram ready to be solved
    """

    biorbd_model = (biorbd.Model(biorbd_model_path), biorbd.Model(biorbd_model_path), biorbd.Model(biorbd_model_path), biorbd.Model(biorbd_model_path),biorbd.Model(biorbd_model_path))

    # Problem parameters
    if long_optim:
        n_shooting = (25, 25, 25, 25)
    else:
        n_shooting = (15, 15, 15, 15, 15)
    final_time = (mean_time_phase_0, mean_time_phase_1, mean_time_phase_2, mean_time_phase_3, mean_time_phase_4)
    tau_min, tau_max, tau_init = -200, 200, 0

    # Add objective functions
    objective_functions = ObjectiveList()
    objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_CONTROL, key="tau", weight=1000, phase=0)
    objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_CONTROL, key="tau", weight=1000, phase=1)
    objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_CONTROL, key="tau", weight=1000, phase=2)
    objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_CONTROL, key="tau", weight=1000, phase=3)
    objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_CONTROL, key="tau", weight=1000, phase=4)

  #  objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_STATE, key="q",weight=100)

    objective_functions.add(
        minimize_difference,
        custom_type=ObjectiveFcn.Mayer,
        node=Node.TRANSITION,
        weight=1000,
        phase=1,
        quadratic=True,
    )


    objective_functions.add(ObjectiveFcn.Mayer.SUPERIMPOSE_MARKERS,
                            custom_type=ObjectiveFcn.Mayer,
                            node=Node.START,
                            first_marker="middle_hand",
                            second_marker="accord_1_haut",
                            weight=10000, phase=0
                            )

    objective_functions.add(ObjectiveFcn.Mayer.SUPERIMPOSE_MARKERS,
                            custom_type=ObjectiveFcn.Mayer,
                            node=Node.END,
                            first_marker="middle_hand",
                            second_marker="accord_2_haut",
                            weight=10000, phase=0
                            )


    objective_functions.add(ObjectiveFcn.Mayer.SUPERIMPOSE_MARKERS,
                            custom_type=ObjectiveFcn.Mayer,
                            node=Node.END,
                            first_marker="middle_hand",
                            second_marker="accord_2_haut",
                            weight=10000, phase=1
                            )

    objective_functions.add(ObjectiveFcn.Mayer.SUPERIMPOSE_MARKERS,
                            custom_type=ObjectiveFcn.Mayer,
                            node=Node.END,
                            first_marker="middle_hand",
                            second_marker="accord_3_haut",
                            weight=10000, phase=2
                            )

    objective_functions.add(ObjectiveFcn.Mayer.SUPERIMPOSE_MARKERS,
                            custom_type=ObjectiveFcn.Mayer,
                            node=Node.END,
                            first_marker="middle_hand",
                            second_marker="accord_3_haut",
                            weight=10000, phase=3
                            )

    objective_functions.add(ObjectiveFcn.Mayer.SUPERIMPOSE_MARKERS,
                            custom_type=ObjectiveFcn.Mayer,
                            node=Node.END,
                            first_marker="middle_hand",
                            second_marker="accord_1_haut",
                            weight=10000, phase=4
                            )


    # Dynamics
    dynamics = DynamicsList()
    expand = False if isinstance(ode_solver, OdeSolver.IRK) else True
    dynamics.add(DynamicsFcn.TORQUE_DRIVEN, expand=expand)
    dynamics.add(DynamicsFcn.TORQUE_DRIVEN, expand=expand)
    dynamics.add(DynamicsFcn.TORQUE_DRIVEN, expand=expand)
    dynamics.add(DynamicsFcn.TORQUE_DRIVEN, expand=expand)
    dynamics.add(DynamicsFcn.TORQUE_DRIVEN, expand=expand)

    # Constraints
    constraints = ConstraintList()
    # Superimpositions
    constraints.add(ConstraintFcn.SUPERIMPOSE_MARKERS, node=Node.START, first_marker="middle_hand", second_marker="accord_1_haut",phase=0)
    constraints.add(ConstraintFcn.SUPERIMPOSE_MARKERS, node=Node.END, first_marker="middle_hand", second_marker="accord_2_haut", phase=0)
    constraints.add(ConstraintFcn.SUPERIMPOSE_MARKERS, node=Node.END, first_marker="middle_hand", second_marker="accord_2_haut", phase=1)
    constraints.add(ConstraintFcn.SUPERIMPOSE_MARKERS, node=Node.END, first_marker="middle_hand", second_marker="accord_3_haut", phase=2)
    constraints.add(ConstraintFcn.SUPERIMPOSE_MARKERS, node=Node.END, first_marker="middle_hand", second_marker="accord_3_haut", phase=3)
    constraints.add(ConstraintFcn.SUPERIMPOSE_MARKERS, node=Node.END, first_marker="middle_hand", second_marker="accord_1_haut", phase=4)




    #Target velocity z , with min bound = mean vel z - stdev and max bound = mean vel z + stdev

    constraints.add(ConstraintFcn.TRACK_MARKERS_VELOCITY, target=vel_z_0, min_bound=-stdev_vel_z_0, max_bound=stdev_vel_z_0, node=Node.START, phase=0, axes=Axis.Z, marker_index=0)


    constraints.add(ConstraintFcn.TRACK_MARKERS_VELOCITY, target=vel_z_1, min_bound=-stdev_vel_z_1, max_bound=stdev_vel_z_1,
                    node=Node.END, phase=0, axes=Axis.Z, marker_index=0)

    constraints.add(ConstraintFcn.TRACK_MARKERS_VELOCITY, target=vel_z_2, min_bound=-stdev_vel_z_2, max_bound=stdev_vel_z_2,
                    node=Node.END, phase=1, axes=Axis.Z, marker_index=0)


    constraints.add(ConstraintFcn.TRACK_MARKERS_VELOCITY, target=vel_z_3, min_bound=-stdev_vel_z_3, max_bound=stdev_vel_z_3,
                    node=Node.END, phase=2, axes=Axis.Z, marker_index=0)

    constraints.add(ConstraintFcn.TRACK_MARKERS_VELOCITY, target=vel_z_4, min_bound=-stdev_vel_z_4, max_bound=stdev_vel_z_4,
                    node=Node.END, phase=3, axes=Axis.Z, marker_index=0)
    constraints.add(ConstraintFcn.TRACK_MARKERS_VELOCITY, target=vel_z_5, min_bound=-stdev_vel_z_5, max_bound=stdev_vel_z_5,
                    node=Node.END, phase=4, axes=Axis.Z, marker_index=0)




    # Path constraint
    x_bounds = BoundsList()
    x_bounds.add(bounds=QAndQDotBounds(biorbd_model[0]))
    x_bounds.add(bounds=QAndQDotBounds(biorbd_model[0]))
    x_bounds.add(bounds=QAndQDotBounds(biorbd_model[0]))
    x_bounds.add(bounds=QAndQDotBounds(biorbd_model[0]))
    x_bounds.add(bounds=QAndQDotBounds(biorbd_model[0]))


    # Initial guess
    x_init = InitialGuessList()
    x_init.add([0] * (biorbd_model[0].nbQ() + biorbd_model[0].nbQdot()))
    x_init.add([0] * (biorbd_model[0].nbQ() + biorbd_model[0].nbQdot()))
    x_init.add([0] * (biorbd_model[0].nbQ() + biorbd_model[0].nbQdot()))
    x_init.add([0] * (biorbd_model[0].nbQ() + biorbd_model[0].nbQdot()))
    x_init.add([0] * (biorbd_model[0].nbQ() + biorbd_model[0].nbQdot()))

    # Define control path constraint
    u_bounds = BoundsList()
    u_bounds.add([tau_min] * biorbd_model[0].nbGeneralizedTorque(), [tau_max] * biorbd_model[0].nbGeneralizedTorque())
    u_bounds.add([tau_min] * biorbd_model[0].nbGeneralizedTorque(), [tau_max] * biorbd_model[0].nbGeneralizedTorque())
    u_bounds.add([tau_min] * biorbd_model[0].nbGeneralizedTorque(), [tau_max] * biorbd_model[0].nbGeneralizedTorque())
    u_bounds.add([tau_min] * biorbd_model[0].nbGeneralizedTorque(), [tau_max] * biorbd_model[0].nbGeneralizedTorque())
    u_bounds.add([tau_min] * biorbd_model[0].nbGeneralizedTorque(), [tau_max] * biorbd_model[0].nbGeneralizedTorque())

    u_init = InitialGuessList()
    u_init.add([tau_init] * biorbd_model[0].nbGeneralizedTorque())
    u_init.add([tau_init] * biorbd_model[0].nbGeneralizedTorque())
    u_init.add([tau_init] * biorbd_model[0].nbGeneralizedTorque())
    u_init.add([tau_init] * biorbd_model[0].nbGeneralizedTorque())
    u_init.add([tau_init] * biorbd_model[0].nbGeneralizedTorque())



    return OptimalControlProgram(
        biorbd_model,
        dynamics,
        n_shooting,
        final_time,
        x_init,
        u_init,
        x_bounds,
        u_bounds,
        objective_functions,
        constraints,
        ode_solver=ode_solver,
      #  phase_transitions=phase_transitions,
    )


def main():
    """
    Defines a multiphase ocp and animate the results
    """

    ocp = prepare_ocp(long_optim=False)

    # --- Solve the program --- #
    solv = Solver.IPOPT(show_online_optim=True)
    solv.set_linear_solver("ma57")
    sol = ocp.solve(solv)
    ocp.print(to_console=False, to_graph=False)
    # --- Show results --- #
    sol.animate()
    sol.print()



if __name__ == "__main__":
    main()
