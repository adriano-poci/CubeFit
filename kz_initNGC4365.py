"""
#!/apps/skylake/software/core/anaconda3/5.1.0/bin/python3
#SBATCH -A oz059
#SBATCH --job-name="slurmSpecNGC4365"
#SBATCH --time=2-00:00
#SBATCH -D "/fred/oz059/poci/muse"
#SBATCH --output="/fred/oz059/poci/muse/slurmSpecNGC4365.log"
#SBATCH --error="/fred/oz059/poci/muse/slurmSpecNGC4365.log"
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=200GB
#SBATCH --mail-type=ALL
#SBATCH --mail-user=adriano.poci@students.mq.edu.au

    slurmSpecNGC4365.py
    Adriano Poci
    Durham University
    2021

    Platforms
    ---------
    Unix, Windows

    Synopsis
    --------
    This module executes some function in the `SLURM` queueing environment

    Authors
    -------
    Adriano Poci <adriano.poci@durham.ac.uk>

History
-------
v1.0:	12 November 2021
"""

# from site import addsitedir as sas
# import pathlib as plp
# sas(str(plp.Path(plp.os.sep, 'fred', 'oz059', 'poci')))
# sas(str(plp.Path(plp.os.sep, 'fred', 'oz059', 'poci', 'dynamics')))
# sas(str(plp.Path(plp.os.sep, 'fred', 'oz059', 'poci', 'pxf')))
# sas(str(plp.Path(plp.os.sep, 'fred', 'oz059', 'poci', 'muse')))
# do not need to add to paths, if run with
#   mpiexec -usize <nProcs+1> -n 1 ipython slurmSpecFCC170.py

# props = dict(galaxy='NGC4365', mPath='hdhdc4365', SN=100, nCuts=393, lOrder=7,
#     specRange=[5100, 5950], full=True, lsf=True, iso='BaSTI', nProcs=1,
#     band='F814W', smask=[[5550, 5560]], genSwitch=None)
# props = dict(galaxy='NGC4365', mPath='hdhdc4365', SN=100, nCuts=207, lOrder=12,
props = dict(galaxy='NGC4365', mPath='hdhdc4365', SN=100, nCuts=3, lOrder=12,
    specRange=[5100, 6650], full=True, lsf=True, iso='BaSTI', nProcs=1,
    band='F814W', genSwitch=None, kind='SMILES', cont=True,
    smask=[[5530, 5555], [6255, 6270], [6320, 6335], [7580, 7700], [8775, 9000]],
    lam=1e-12)
