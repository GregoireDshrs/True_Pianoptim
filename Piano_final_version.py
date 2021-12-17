"Pianoptim "
import biorbd_casadi as biorbd
import numpy as np
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
)


# Constants from data collect
# mean positions and mean velocities of the right hand of subject 134 for 5 events, calculated for 30 cycles
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

stdev_vel_y_0 = 0.06590577
stdev_vel_y_1 = 0.04433499
stdev_vel_y_2 = 0.08251966
stdev_vel_y_3 = 0.03813032
stdev_vel_y_4 = 0.07607116
stdev_vel_y_5 = 0.0713205

stdev_vel_z_0 = 0.11591871
stdev_vel_z_1 = 0.10771169
stdev_vel_z_2 = 0.081717
stdev_vel_z_3 = 0.09894744
stdev_vel_z_4 = 0.11820802
stdev_vel_z_5 = 0.1479469

mean_time_phase_0 = 0.36574653            # phase 0 is from first marker to second marker
phase_appui = 0.16                       # time corresponding to a keystroke (in second)
mean_time_phase_0_bis = mean_time_phase_0-phase_appui
mean_time_phase_1 = 0.10555556
mean_time_phase_1_bis = mean_time_phase_1-phase_appui
mean_time_phase_2 = 0.40625
mean_time_phase_2_bis = mean_time_phase_2-phase_appui
mean_time_phase_3 = 0.10387153
mean_time_phase_3_bis = mean_time_phase_3-phase_appui
mean_time_phase_4 = 1.00338542            # phase 4 is from last marker to first marker
mean_time_phase_4_bis = mean_time_phase_4-phase_appui



def minimize_difference(all_pn: PenaltyNode):
    return all_pn[0].nlp.controls.cx_end - all_pn[1].nlp.controls.cx



def prepare_ocp(
        biorbd_model_path: str = "Piano_final_version.bioMod",
        ode_solver: OdeSolver = OdeSolver.RK8(),
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
     # there are as many biorbd.Model as there are phases
    biorbd_model = (biorbd.Model(biorbd_model_path),
                    biorbd.Model(biorbd_model_path),
                    biorbd.Model(biorbd_model_path),
                    biorbd.Model(biorbd_model_path),
                    biorbd.Model(biorbd_model_path),
                    biorbd.Model(biorbd_model_path),
                    biorbd.Model(biorbd_model_path),
                    biorbd.Model(biorbd_model_path),
                    biorbd.Model(biorbd_model_path),
                    biorbd.Model(biorbd_model_path))



    n_shooting = (20,16,20,16,20,16,20,16,20,16)    # arbitrary choices
    final_time = (0.08,0.33,0.08,0.08,0.08,0.37,0.08,0.08,0.08,1)    # TODO recheck time phases, use mean_time and phase_appui
    tau_min, tau_max, tau_init = -1000, 1000, 0

    # Add objective functions
    objective_functions = ObjectiveList()
    objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_CONTROL, key="tau", weight=100,phase=0)
    objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_STATE, key="q", weight=100,phase=0)
    objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_CONTROL, key="tau", weight=100, phase=1)
    objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_STATE, key="q", weight=1000, phase=1)
    objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_CONTROL, key="tau", weight=100, phase=2)
    objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_STATE, key="q", weight=1000, phase=2)
    objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_CONTROL, key="tau", weight=100, phase=3)
    objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_STATE, key="q", weight=1000, phase=3)
    objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_CONTROL, key="tau", weight=100, phase=4)
    objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_STATE, key="q", weight=1000, phase=4)
    objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_CONTROL, key="tau", weight=100, phase=5)
    objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_STATE, key="q", weight=1000, phase=5)
    objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_CONTROL, key="tau", weight=100, phase=6)
    objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_STATE, key="q", weight=1000, phase=6)
    objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_CONTROL, key="tau", weight=100, phase=7)
    objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_STATE, key="q", weight=1000, phase=7)
    objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_CONTROL, key="tau", weight=100, phase=8)
    objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_STATE, key="q", weight=1000, phase=8)
    objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_CONTROL, key="tau", weight=100, phase=9)
    objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_STATE, key="q", weight=1000, phase=9)

    # minimize the difference between phases
    objective_functions.add(
        minimize_difference,
        custom_type=ObjectiveFcn.Mayer,
        node=Node.TRANSITION,
        weight=1000,
        quadratic=True,
    )

    # The following objective functions are all SUPERIMPOSE_MARKERS and aim at leading the hand to 'play' the 3 chords
    objective_functions.add(ObjectiveFcn.Mayer.SUPERIMPOSE_MARKERS,
                            custom_type=ObjectiveFcn.Mayer,
                            node=Node.START,
                            first_marker="middle_hand",
                            second_marker="accord_1_haut",
                            weight=10000, phase=0
                            )
    objective_functions.add(ObjectiveFcn.Mayer.SUPERIMPOSE_MARKERS,
                            custom_type=ObjectiveFcn.Mayer,
                            node=Node.MID,
                            first_marker="middle_hand",
                            second_marker="accord_1_bas",
                            weight=10000, phase=0
                            )
    objective_functions.add(ObjectiveFcn.Mayer.SUPERIMPOSE_MARKERS,
                            custom_type=ObjectiveFcn.Mayer,
                            node=Node.END,
                            first_marker="middle_hand",
                            second_marker="accord_1_haut",
                            weight=10000, phase=0
                            )

    objective_functions.add(ObjectiveFcn.Mayer.SUPERIMPOSE_MARKERS,
                            custom_type=ObjectiveFcn.Mayer,
                            node=Node.MID,
                            first_marker="middle_hand",
                            second_marker="m_inter0",
                            weight=10000, phase=1
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
                            node=Node.MID,
                            first_marker="middle_hand",
                            second_marker="accord_2_bas",
                            weight=10000, phase=2
                            )
    objective_functions.add(ObjectiveFcn.Mayer.SUPERIMPOSE_MARKERS,
                            custom_type=ObjectiveFcn.Mayer,
                            node=Node.END,
                            first_marker="middle_hand",
                            second_marker="accord_2_haut",
                            weight=10000, phase=2
                            )

    objective_functions.add(ObjectiveFcn.Mayer.SUPERIMPOSE_MARKERS,
                            custom_type=ObjectiveFcn.Mayer,
                            node=Node.END,
                            first_marker="middle_hand",
                            second_marker="accord_2_haut",
                            weight=10000, phase=3
                            )
    objective_functions.add(ObjectiveFcn.Mayer.SUPERIMPOSE_MARKERS,
                            custom_type=ObjectiveFcn.Mayer,
                            node=Node.MID,
                            first_marker="middle_hand",
                            second_marker="accord_2_bas",
                            weight=10000, phase=4
                            )
    objective_functions.add(ObjectiveFcn.Mayer.SUPERIMPOSE_MARKERS,
                            custom_type=ObjectiveFcn.Mayer,
                            node=Node.END,
                            first_marker="middle_hand",
                            second_marker="accord_2_haut",
                            weight=10000, phase=4
                            )
    objective_functions.add(ObjectiveFcn.Mayer.SUPERIMPOSE_MARKERS,
                            custom_type=ObjectiveFcn.Mayer,
                            node=Node.MID,
                            first_marker="middle_hand",
                            second_marker="m_inter2",
                            weight=10000, phase=5
                            )
    objective_functions.add(ObjectiveFcn.Mayer.SUPERIMPOSE_MARKERS,
                            custom_type=ObjectiveFcn.Mayer,
                            node=Node.END,
                            first_marker="middle_hand",
                            second_marker="accord_3_haut",
                            weight=10000, phase=5
                            )

    objective_functions.add(ObjectiveFcn.Mayer.SUPERIMPOSE_MARKERS,
                            custom_type=ObjectiveFcn.Mayer,
                            node=Node.MID,
                            first_marker="middle_hand",
                            second_marker="accord_3_bas",
                            weight=10000, phase=6
                            )
    objective_functions.add(ObjectiveFcn.Mayer.SUPERIMPOSE_MARKERS,
                            custom_type=ObjectiveFcn.Mayer,
                            node=Node.END,
                            first_marker="middle_hand",
                            second_marker="accord_3_haut",
                            weight=10000, phase=6
                            )

    objective_functions.add(ObjectiveFcn.Mayer.SUPERIMPOSE_MARKERS,
                            custom_type=ObjectiveFcn.Mayer,
                            node=Node.END,
                            first_marker="middle_hand",
                            second_marker="accord_3_haut",
                            weight=10000, phase=7
                            )
    objective_functions.add(ObjectiveFcn.Mayer.SUPERIMPOSE_MARKERS,
                            custom_type=ObjectiveFcn.Mayer,
                            node=Node.MID,
                            first_marker="middle_hand",
                            second_marker="accord_3_bas",
                            weight=10000, phase=8
                            )
    objective_functions.add(ObjectiveFcn.Mayer.SUPERIMPOSE_MARKERS,
                            custom_type=ObjectiveFcn.Mayer,
                            node=Node.END,
                            first_marker="middle_hand",
                            second_marker="accord_3_haut",
                            weight=10000, phase=8
                            )
    objective_functions.add(ObjectiveFcn.Mayer.SUPERIMPOSE_MARKERS,
                            custom_type=ObjectiveFcn.Mayer,
                            node=Node.MID,
                            first_marker="middle_hand",
                            second_marker="m_inter2",
                            weight=10000, phase=9
                            )
    objective_functions.add(ObjectiveFcn.Mayer.SUPERIMPOSE_MARKERS,
                            custom_type=ObjectiveFcn.Mayer,
                            node=Node.END,
                            first_marker="middle_hand",
                            second_marker="accord_1_haut",
                            weight=10000, phase=9
                            )

    # Dynamics
    dynamics = DynamicsList()
    for i in range(len(biorbd_model)):
        dynamics.add(DynamicsFcn.TORQUE_DRIVEN)  # there are no muscles, so it is torque driven

    # Constraints
    constraints = ConstraintList()
    # Superimpositions
    constraints.add(ConstraintFcn.SUPERIMPOSE_MARKERS, node=Node.START, first_marker="middle_hand",
                    second_marker="accord_1_haut", phase=0)
    constraints.add(ConstraintFcn.SUPERIMPOSE_MARKERS, node=Node.MID, first_marker="middle_hand",
                    second_marker="accord_1_bas", phase=0)
    constraints.add(ConstraintFcn.SUPERIMPOSE_MARKERS, node=Node.END, first_marker="middle_hand",
                    second_marker="accord_1_haut", phase=0)
    constraints.add(ConstraintFcn.SUPERIMPOSE_MARKERS, node=Node.MID, first_marker="middle_hand",
                    second_marker="m_inter0", phase=1)
    constraints.add(ConstraintFcn.SUPERIMPOSE_MARKERS, node=Node.END, first_marker="middle_hand",
                    second_marker="accord_2_haut", phase=1)
    constraints.add(ConstraintFcn.SUPERIMPOSE_MARKERS, node=Node.MID, first_marker="middle_hand",
                    second_marker="accord_2_bas", phase=2)
    constraints.add(ConstraintFcn.SUPERIMPOSE_MARKERS, node=Node.END, first_marker="middle_hand",
                    second_marker="accord_2_haut", phase=2)
    constraints.add(ConstraintFcn.SUPERIMPOSE_MARKERS, node=Node.END, first_marker="middle_hand",
                    second_marker="accord_2_haut", phase=3)
    constraints.add(ConstraintFcn.SUPERIMPOSE_MARKERS, node=Node.MID, first_marker="middle_hand",
                    second_marker="accord_2_bas", phase=4)
    constraints.add(ConstraintFcn.SUPERIMPOSE_MARKERS, node=Node.END, first_marker="middle_hand",
                    second_marker="accord_2_haut", phase=4)
    constraints.add(ConstraintFcn.SUPERIMPOSE_MARKERS, node=Node.MID, first_marker="middle_hand",
                    second_marker="m_inter2", phase=5)
    constraints.add(ConstraintFcn.SUPERIMPOSE_MARKERS, node=Node.END, first_marker="middle_hand",
                    second_marker="accord_3_haut", phase=5)
    constraints.add(ConstraintFcn.SUPERIMPOSE_MARKERS, node=Node.MID, first_marker="middle_hand",
                    second_marker="accord_3_bas", phase=6)
    constraints.add(ConstraintFcn.SUPERIMPOSE_MARKERS, node=Node.END, first_marker="middle_hand",
                    second_marker="accord_3_haut", phase=6)
    constraints.add(ConstraintFcn.SUPERIMPOSE_MARKERS, node=Node.END, first_marker="middle_hand",
                    second_marker="accord_3_haut", phase=7)
    constraints.add(ConstraintFcn.SUPERIMPOSE_MARKERS, node=Node.MID, first_marker="middle_hand",
                    second_marker="accord_3_bas", phase=8)
    constraints.add(ConstraintFcn.SUPERIMPOSE_MARKERS, node=Node.END, first_marker="middle_hand",
                    second_marker="accord_3_haut", phase=8)
    constraints.add(ConstraintFcn.SUPERIMPOSE_MARKERS, node=Node.MID, first_marker="middle_hand",
                    second_marker="m_inter1", phase=9)
    constraints.add(ConstraintFcn.SUPERIMPOSE_MARKERS, node=Node.END, first_marker="middle_hand",
                    second_marker="accord_1_haut", phase=9)
    constraints.add(ConstraintFcn.TRACK_MARKERS, min_bound=0, node=Node.ALL, axes=Axis.Z, marker_index=0)

    # Constraints on the velocity z of the hand (marker middle_hand has index 0):

    constraints.add(ConstraintFcn.TRACK_MARKERS_VELOCITY, target=vel_z_0, min_bound=-stdev_vel_z_0,
                    max_bound=stdev_vel_z_0, node=Node.MID, phase=0, axes=Axis.Z, marker_index=0)
    constraints.add(ConstraintFcn.TRACK_MARKERS_VELOCITY, target=vel_z_1, min_bound=-stdev_vel_z_1,
                    max_bound=stdev_vel_z_1,
                    node=Node.MID, phase=2, axes=Axis.Z, marker_index=0)
    constraints.add(ConstraintFcn.TRACK_MARKERS_VELOCITY, target=vel_z_2, min_bound=-stdev_vel_z_2,
                    max_bound=stdev_vel_z_2,
                    node=Node.MID, phase=4, axes=Axis.Z, marker_index=0)
    constraints.add(ConstraintFcn.TRACK_MARKERS_VELOCITY, target=vel_z_3, min_bound=-stdev_vel_z_3,
                    max_bound=stdev_vel_z_3,
                    node=Node.MID, phase=6, axes=Axis.Z, marker_index=0)
    constraints.add(ConstraintFcn.TRACK_MARKERS_VELOCITY, target=vel_z_4, min_bound=-stdev_vel_z_4,
                    max_bound=stdev_vel_z_4,
                    node=Node.MID, phase=8, axes=Axis.Z, marker_index=0)
    constraints.add(ConstraintFcn.TRACK_MARKERS_VELOCITY, target=vel_z_5, min_bound=-stdev_vel_z_5,
                    max_bound=stdev_vel_z_5,
                    node=Node.END, phase=9, axes=Axis.Z, marker_index=0)

    # External forces
    # External forces  ( a chord played fortissimo can lead to a maximal force of 30N )
    f0 = np.array([0, 0, 0, 0, 0, 0])
    f1 = np.array([0, -5.1, 0, 0, 0, 30])
    f2 = np.array([-4.47, -5.1, 0, 0, 0, 30])
    f3 = np.array([-8.7, -5.1, 0, 0, 0, 30])

    fnulle = np.array([0, 0, 0, 0, 0, 0])
    fvide = np.linspace(fnulle[:, np.newaxis], fnulle[:, np.newaxis], 16, axis=2)
    # The force is applied linearly during the phase
    flinup1 = np.linspace(f0[:, np.newaxis], f1[:, np.newaxis], 10, axis=2)
    flindown1 = np.linspace(f1[:, np.newaxis], f0[:, np.newaxis], 10, axis=2)
    ftotale1 = np.concatenate((flinup1, flindown1), axis=2)

    flinup2 = np.linspace(f0[:, np.newaxis], f2[:, np.newaxis], 10, axis=2)
    flindown2 = np.linspace(f2[:, np.newaxis], f0[:, np.newaxis], 10, axis=2)
    ftotale2 = np.concatenate((flinup2, flindown2), axis=2)

    flinup3 = np.linspace(f0[:, np.newaxis], f3[:, np.newaxis], 10, axis=2)
    flindown3 = np.linspace(f3[:, np.newaxis], f0[:, np.newaxis], 10, axis=2)
    ftotale3 = np.concatenate((flinup3, flindown3), axis=2)

    fext = [ftotale1, fvide, ftotale2, fvide, ftotale2, fvide, ftotale3, fvide, ftotale3, fvide]

    # Path constraint
    x_bounds = BoundsList()
    for j in range(len(biorbd_model)):
        x_bounds.add(bounds=QAndQDotBounds(biorbd_model[0]))

    # Initial guess
    x_init = InitialGuessList()
    for k in range(len(biorbd_model)):
        x_init.add([0] * (biorbd_model[0].nbQ() + biorbd_model[0].nbQdot()))

        # Define control path constraint
    u_bounds = BoundsList()
    for l in range(len(biorbd_model)):
        u_bounds.add([tau_min] * biorbd_model[0].nbGeneralizedTorque(),
                     [tau_max] * biorbd_model[0].nbGeneralizedTorque())

    u_init = InitialGuessList()
    for m in range(len(biorbd_model)):
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
        objective_functions=objective_functions,
        constraints=constraints,
        ode_solver=ode_solver,
        external_forces=fext,
    )


def main():

    ocp = prepare_ocp()

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

