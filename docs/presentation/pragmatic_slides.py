#!/usr/bin/env python
# -*- coding: latin-1 -*-

from latexslides import *

# Set institutions
inst1, inst2 = "Imperial College London", "Fujitsu Laboratories of Europe"

# Set authors
authors = [("Gerard Gorman", inst1),
           ("James Southern", inst2),
           ("Patrick Farrell", inst1),
           ("Matthew Piggott", inst1)]

# Create slides, exchange 'Beamer' with 'Prosper' for Prosper
slides = BeamerSlides(title="PRAgMaTIc: Parallel anisotRopic Adaptive Mesh ToolkIt",
                      author_and_inst=authors,
                      toc_heading=None)

collection = []

collection = collection + [Section("Introduction")]

collection = collection + [
    BulletSlide(title="Aims",
                # block_heading="Aims",
                bullets=["Enable multiscale modelling in complex domains.",
                         "Provide an easy to use anisotropic adaptive mesh toolkit:",
                         ["2D triangular meshes.",
                          "3D tetrahedral meshes.",
                          "OpenMP parallelism within multicore nodes.",
                          "MPI parallelism between distributed memory nodes."],
                         "Open source development model:",
                         ["Collaborative development: https://launchpad.net",
                          "Desktop: Packages distributed for Ubuntu.",
                          "Clusters/Supercomputers: OPL - http://www.openpetascale.org/"]],
                hidden=False)
    ]

collection = collection + [
    BulletSlide(title="Features",
                # block_heading="Objectives",
                bullets=["Control solution errors through element size.",
                         ["Ideally error metrics come from application code.",
                          "Blackbox p-norm error estimator provided."],
                         "Mesh edge refinement/coarsening.",
                         "Element swapping.",
                         "Optimisation based mesh smoothing.",
                         "Surface operations are currently confined to coplanar patches."],
                hidden=False)
    ]

collection = collection + [
    BulletSlide(title="Infrastructure",
                # block_heading="Infrastructure",
                bullets=["Software repository and collaborative environment supplied by launchpad.",
                         ["Make it easy for developers to get involved."],
                         "Daily GCC builds on Ubuntu releases:",
                         ["Natty, Oneiric and Precise."],
                         "OPL/Buildbot:",
                         ["Nightly Intel build for release and trunk.",
                          "Executes test matrix."],
                         "Users manual.",
                         "Source code documentation with dOxygen."],
                hidden=False)
    ]

collection = collection + [
    Section("Adaptivity"),
    
    BulletSlide(title="Key steps",
                # block_heading="class Mesh",
                bullets=["Error evaluation.",
                         "Element size specification.",
                         "Loop:",
                         ["Mesh coarsening and refinement to tune local element size.",
                          "Element/edge swaps to improve quality."],
                         "Optimisation based smoothing to finesse mesh quality.",
                         "Load-balancing.",
                         "Interpolation of solution fields between old and new mesh."],
                hidden=False),

    BulletSlide(title="Error measures",
                # block_heading="class Mesh",
                bullets=["Ideally error estimates should be supplied by the application code --- dependent upon:",
                         ["Current state of the system.",
                          "Discretication of the equations being solved.",
                          "Boundary conditions"],
                         "Blackbox p-norm provided which can be applied purely to scalar fields",
                         ],
                hidden=False),
    
    Slide(title="$L_p$ norm of the error",
          content=[TextBlock(r"""
\begin{equation}\label{eq:metric_l2}
M_p({\bf x}) = \frac{1}{\epsilon({\bf x})}(\det(|H({\bf
x})|))^{-\frac{1}{2p+n}}|H({\bf x})| = (\det(|H({\bf
x})|))^{-\frac{1}{2p+n}}M_\infty \, ,
\end{equation}

where $n$ is the dimension of the space and $p \in \mathbb{Z}^+$. With
this metric, the $L_p$ norm of the interpolation error is bounded by

\begin{equation}
||f({\bf x})-\delta({\bf x})||_p = \left( \int_\Omega |f({\bf
x})-\delta({\bf x})|^{p} \right) ^{1/p} \leq CN^{-2/n}||^n \sqrt{\det
\, |H({\bf x})|}||_{\frac{pn}{2p+n}(\Omega)} \, ,
\end{equation}

where $N$ is the number of elements in the mesh. Note that in the limit \mbox{$p \to
\infty\, , \, (\det(|H|))^{-\frac{1}{2p+n}} \to 1$} and the absolute
metric,  $M_\infty$, is recovered.
""")],
          ),
    ]

collection = collection + [
    Section("NUMA considerations"),
    
    Slide(title="OpenMP/NUMA",
          content=[TextBlock(r"""
Non-Uniform Memory Architecture (NUMA) is a computer memory design used in multiprocessors, where the memory access time depends on the memory location relative to a processor. Under NUMA, a processor can access its own local memory faster than non-local memory, that is, memory local to another processor or memory shared between processors.""")],
          figure='NUMA.pdf',
          figure_fraction_width=0.7,),
    
    Slide("Intel Core Westmere processor",
          content=[TextBlock(r"""
Xeon CPU X5650 @ 2.67GHz""")],
          figure='westmere.pdf',
          figure_fraction_width=1.2,),
    
    BulletSlide(title="Memory binding",
                bullets=["It is important memory required by a thread is allocated close to the core the thread is associated with",
                         "By default, page faults are satisfied by memory from the node containing the page-faulting CPU.",
                         "Because the first CPU to touch the page will be the CPU that faults the page in, this default policy is called \"first touch\""],
                block_heading="First touch",
                hidden=False),
    
    BulletSlide(title="Processor affinity",
                bullets=["Native queue scheduling algorithm not optimal for HPC",
                         "Processor affinity is a modification of the native scheduling algorithm",
                         "Each thread in the queue has a tag indicating its preferred core",
                         "Processor affinity takes advantage of the fact that some remnants of a process may remain in one processor's state (in particular, in its cache) from the last time the process ran, and so scheduling it to run on the same processor the next time could result in the process running more efficiently than if it were to run on another processor. Overall system efficiency increases by reducing performance-degrading situations such as cache misses"],
                block_heading="",
                hidden=False),
    
    Slide("Good Vs bad memory mapping",
          content=[TextBlock(r"""
Benchmark by Hongzhang Shan (Lawrence Berkeley National Laboratory) compute nodes of dual twelve-core AMD MagnyCours: Red curve shows performance when all memory is bound to a single memory node.  Blue curve shows corrected memory initialization using first-touch so that thread only accesses data on local memory controller (the correct NUMA mapping).""")],
          figure='bandwidth.pdf',
          figure_fraction_width=0.5,),
    
    Slide("Sparc64 VIIIfx and the K-computer",
          content=[TextBlock(r"""
Non-uniform memory access might not be a significant factor: integrated memory controller greatly reduces memory latency.""")],
          figure='SPARC64_VIIIfx.pdf',
          figure_fraction_width=0.9,)
    ]

collection = collection + [
    Section("Results"),
    
    Slide("High magnitude shock",
          content=[TextBlock(r"""Consider a field described by:
\begin{equation}
\forall (x, y) \in [0, 1]^2, f(x, y) = 0.1 \sin(50x) + \arctan\left(\frac{0.1}{\sin(5y)-2x}\right).
\end{equation}
The high amplitude shock is induced by the $\arctan$ function and
small variations in amplitude are superimposed with a sine function. The L1 error norm was used to
form the metric tensor field as it is more effective at capturing
multiscale features (Loseille 2009).""")]),

    Slide("High magnitude shock",
          # content=[TextBlock(r"""mesh - scalar field.""")],
          figure='adapt_shock.png',
          figure_pos='w', figure_fraction_width=1.0, left_column_width=1.0,),

    Slide("High magnitude shock",
          content=[TextBlock(r"""Element quality before adaptivity:
Mean:    0.76
Minimum: 0.027
RMS:     0.16

Element quality after adaptivity:
Mean:    0.91
Minimum: 0.66
RMS:     0.056
""")],)
    ]

collection = collection + [
    Section("Roadmap"),

    BulletSlide(title="Future work",
                # block_heading="Additional features",
                bullets=["Benchmarking on K-Computer test cluster.",
                         "Benchmarking OpenMP in Intel MIC.",
                         "Support for curved boundaries.",
                         "Support for prisms on boundaries --- important for viscous boundary layers.",
                         "CUDA/OpenCL implementation.",
                         "Application:",
                         ["Fluidity: range of geophysical, industrial flows, solid mechanics, radiation transport.",
                          "NEKTAR++: Biomedical flows",
                          "Invite other HPC application developers to use/collaborate."], 
                         ],
                hidden=False)]

slides.add_slides(collection)

# Dump to file:
slides.write("pragmatic_slides.tex")
