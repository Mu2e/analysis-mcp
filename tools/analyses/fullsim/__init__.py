"""Full-simulation analyses: reconstructed EventNtuple files with MC truth.

Modules:
  eventntuple.py   read EventNtuple/ntuple in pyfitter's groups; the MC
                   origin of each event's first track (tools/eventntuple.py)
  cuts.py          pyfitter's cut-set 80 CE-like track cuts
  normalization.py stopped mu- per POT and rates per stopped mu-, as Production
  provenance.py    events generated to make a file, from SAM dh.gencount
  sensitivity.py   fullsim_sensitivity (SPEC): S/sqrt(B) as approx_ce_sensitivity, from CE mix

README.md here says how to run them.
"""
