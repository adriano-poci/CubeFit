#!/usr/bin/env python3
"""
    kz_init.py
    Adriano Poci
    University of Oxford
    2026

    <adriano.poci@physics.ox.ac.uk>

    Platforms
    ---------
    Unix, Windows

    Synopsis
    --------
    Light Python wrapper defining per-galaxy properties.

    Author
    ------
    Adriano Poci <adriano.poci@physics.ox.ac.uk>


History
-------
v1.0:	Made universal `kz_init` for all galaxies. 22 July 2026
"""

def props(galaxy):
    if 'NGC4365' in galaxy:
        propDict = dict(galaxy='NGC4365', mPath='hdhdc4365', SN=100, nCuts=154,
            lOrder=0, specRange=[5100, 6650], full=True, lsf=True, iso='BaSTI',
            nProcs=1, band='F814W', genSwitch=None, kind='SMILES', cont=False,
            smask=[[5530, 5555], [6255, 6335], [7580, 7700], [8775, 9000]],
            warm='resume')
    elif 'FCC170' in galaxy:
        propDict = dict(galaxy='FCC170', mPath='hd170', SN=100, nCuts=3,
            lOrder=0, specRange=[5100, 6650], full=True, lsf=True, iso='BaSTI',
            nProcs=1, band='r', genSwitch=None, kind='SMILES', cont=False,
            smask=[[5530, 5555], [6255, 6335], [7580, 7700], [8775, 9000]],
            warm='resume',
            sspIdx=([-1.5, -1.0, -0.6, -0.3, 0.0, 0.15, 0.26, 0.4],
            [3.0, 6.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0],
            [-0.2, 0.0, 0.2, 0.4, 0.6]))
    else:
        raise ValueError(f"Unknown galaxy '{galaxy}'")
    
    return propDict