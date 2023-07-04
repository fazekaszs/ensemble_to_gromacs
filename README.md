# ensemble_to_gromacs
These scripts can be used to analyse the Phi and Psi dihedral (or torsion) angle distribution in a protein structural ensemble (e.g. from Chemical Shift-Rosetta), and to define potential energy functions (PEFs), to exchange them for the original torsional energy terms in GROMACS for a molecular dynamisc (MD) simulation.

1. You can adjust your settings in the "config.py" file first. (Detailed description will be available later.)
2. The dihedral angles in your ensemble can be measured and saved to a pickle using "save_dihedrals.py".
3. Running the "visualize_dihedrals.py" you can prepare figures about the dihedral angle distribution for every residue.
4. In the next step you should run "fit_dihedrals.py", where you define a probability density function (PDF) according to the dihedral angle distributions using kernel density estimation. After that the script defines you the PEFs.
5. You can look at the angle distributions and the PEFs in case of every residue, by running "visualize_pef.py".
6. Before the final step, you should set up the MD simulation, which you want to run to refine your structural models. You need to have a ".gro" file and a ".top" file about your solvated system, ready to start the simulation. By running the "create_tables.py" you will get a ".new.top" file, which you should use as a topology file for your GROMACS MD simulation.
