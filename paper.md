---
title: 'COMPACT: Comprehensive Prefetching Algorithm Framework'
authors:
  - name: Mahdi Alinejad[^1]
    affiliation: '1'
  - name: Sahand Zoufan[^1]
    affiliation: '1'
  - name: Atiyeh Gheibi-Fetrat
    affiliation: '1'
  - name: Shaahin Hessabi
    affiliation: '1'
  - name: Hamid Sarbazi-Azad
    affiliation: '1,2'
affiliations:
  - index: 1
    name: Sharif University of Technology, Iran
  - index: 2
    name: IPM Institute For Research In Fundamental Sciences, Iran
date: 3 November 2025
bibliography: paper.bib
---

# Summary

COMPACT (Comprehensive Prefetching Algorithm Framework) is a Python-based simulation framework designed to support the rapid development and evaluation of hardware prefetching algorithms. It offers a modular and extensible environment that allows researchers and engineers to explore and test new prefetching techniques without requiring complex hardware changes. The framework includes several common prefetching algorithms, a flexible cache simulator, and built-in tools for analyzing the prefetcher performance. COMPACT is designed for ease of use and customization, making it a practical and valuable resource for the computer architecture research community.

# Statement of Need

As the performance gap between processors and main memory continues to grow, hardware prefetching has become a key approach to improving overall system performance. However, developing and evaluating new prefetching algorithms often requires significant time and resources. COMPACT helps address this issue by providing a lightweight and adaptable simulation framework that allows researchers to efficiently test and compare prefetching strategies. By simplifying the evaluation process, COMPACT supports faster experimentation and innovation in the study of hardware prefetching.

# State of the Field

Several tools exist for research and evaluation of prefetching algorithm. ChampSim [@champsim] is a trace-driven simulator widely used in prefetching competitions, providing infrastructure for evaluating cache replacement and prefetching policies but requiring C++ implementations tightly coupled with simulator internals. Gem5 [@gem5] offers comprehensive cycle-accurate system simulation with extensive microarchitectural detail, but its complexity creates a steep learning curve and long development cycles for prefetcher prototyping. Sniper [@sniper] and ZSim [@zsim] provide faster multi-core simulation with prefetching support, yet they similarly embed prefetcher logic within complex simulator architectures. DRAMSim [@dramsim] focuses on memory system modeling but lacks the prefetcher abstraction layer needed for algorithm comparison.

COMPACT is built as a standalone framework rather than contributing to existing simulators for several reasons. First, COMPACT prioritizes rapid prototyping and algorithm comparison through a clean Python-based interface that decouples prefetcher logic from simulator internals. Second, COMPACT fills a specific niche between educational demonstration tools and production simulators: it offers sufficient fidelity for early-stage research and comparative studies while maintaining the development speed and experimental flexibility that heavyweight simulators cannot provide. This design choice enables researchers to iterate quickly on new ideas and systematically compare algorithms under controlled conditions before investing in detailed hardware validation.

# Software Design

COMPACT is designed as a modular simulation framework that separates prefetching logic from cache modeling, memory access handling, and evaluation metrics, facilitating easy integration of new algorithms and memory access traces. The framework adopts a layered architecture where each prefetcher implements a unified interface, enabling algorithms with fundamentally different prediction mechanisms—temporal, spatial, correlation-based, and learning-based—to be evaluated under identical simulation conditions. This design enables fair comparison across heterogeneous prefetching paradigms without requiring algorithm-specific modifications to the simulator core.

A key design trade-off is prioritizing extensibility and rapid experimentation over cycle-accurate microarchitectural detail. COMPACT favors a lightweight Python-based implementation to reduce development friction and enable fast prototyping, at the cost of lower execution speed compared to full-system simulators like Gem5 or ChampSim. This trade-off is intentional: the framework targets early-stage algorithm design, comparative analysis, and educational use rather than final hardware validation. The modular architecture allows researchers to swap cache configurations, replacement policies, and prefetching algorithms independently through YAML-based configuration files, supporting reproducible experiments without code modification. This abstraction boundary matters because it enables systematic exploration of the prefetching design space while maintaining experimental consistency across diverse algorithm families.

# Research Impact Statement

COMPACT enables systematic and reproducible evaluation of hardware data prefetching algorithms across a wide range of prediction paradigms within a single unified framework. The software provides 33 implemented prefetching algorithms spanning classical heuristics such as Best-Offset [@bestoffset] and GHB [@ghb], spatial predictors such as SMS [@sms] and Bingo [@bingo], temporal correlators such as Domino [@domino] and EBCP [@ebcp], and learning-based approaches such as LSTM-based LearnPrefetch [@hashemi2018], Neural [@neural], and the reinforcement-learning prefetcher Pythia [@pythia], representing the most comprehensive open-source collection available for comparative prefetching research. By standardizing algorithm interfaces, cache modeling, and performance metrics, the framework allows researchers to compare prefetching strategies under identical experimental conditions—a capability difficult to achieve using traditional simulator-based workflows.

The framework is ready for community adoption and demonstrates research significance through key professional features: a comprehensive automated test suite covering all 33 algorithms, structured documentation with detailed API references, a reproducible YAML-based configuration system with sample configurations for major algorithms, open-source licensing, and standard Python packaging for straightforward installation. These characteristics lower the barrier to prefetcher research by reducing implementation overhead, allowing new algorithms to be expressed independently of simulator internals while supporting rapid hypothesis testing and controlled comparisons.

COMPACT fills a documented gap between educational demonstration tools and production simulators, providing sufficient fidelity for early-stage algorithm design and comparative studies while maintaining development speed that heavyweight simulators cannot match. This feature positions COMPACT as a practical foundation for ongoing computer architecture research, and as an educational platform enabling students to implement and analyze prefetching mechanisms without extensive simulator expertise.

# AI Usage Disclosure

Generative AI tools are used during the development of this project. AI assistance is primarily employed to generate initial code and test implementations based on detailed algorithm descriptions from the original publications. We review and validate all generated tests for correctness and manually verify that the implemented algorithms match the behavior described in the corresponding papers. AI tools are also used to standardize formatting and phrasing of comments and documentation within the manuscript. We retain full responsibility for the design, correctness, and integrity of the software and the accompanying paper.

# Implemented Algorithms

The following prefetching algorithms are implemented in COMPACT. Brief descriptions are provided for each algorithm based on their original publications.

- **Markov Predictor Prefetcher** [@markovpredictor]: The Markov predictor models memory access behavior by representing addresses as states in a Markov chain. Transition probabilities between states are learned from historical access patterns. During execution, the prefetcher predicts the most likely successor addresses based on the current state (accessed address), issuing prefetches for these predicted next addresses.

- **Correlation Prefetcher** [@correlation]: The correlation prefetcher uses a helper memory thread that observes sequences of cache misses and learns correlations between them. It records recurring temporal miss patterns and uses these correlations to predict future memory accesses. When the thread detects a similar pattern, it replays the correlated sequence of misses to prefetch data ahead of demand accesses.

- **HDS Prefetcher** [@hds]: The dynamic Hot Data Stream (HDS) prefetcher identifies frequently accessed memory streams by profiling bursty access patterns. These "hot" streams are modeled using Deterministic Finite State Machines (DFSMs). During profiling, the most frequent sequences of accesses are encoded in DFSMs. At runtime, the prefetcher predicts future accesses by following the DFSM states, prefetching data ahead of demand accesses.

- **TCP Prefetcher** [@tcp]: The Tag correlating prefetcher operates by correlating cache line tags rather than full memory addresses. It maintains a history of recently accessed cache line tags and predicts the next tag delta based on these correlations. By chaining the predicted tags, TCP can prefetch the next set of cache lines ahead of the demand access. This technique focuses on the correlation of cache line tags, improving prefetch efficiency by predicting multiple cache lines at once.


- **GHB Prefetcher** [@ghb]: The global history buffer prefetcher maintains a Global History Buffer (GHB) of recent memory accesses and correlates the address deltas (the differences between consecutive memory addresses) across all Program Counters (PCs). It tracks the history of deltas and uses this information to predict future memory addresses based on recurring delta patterns. These predictions help issue prefetches for future memory accesses before they occur.

- **Store-Ordered Streamer Prefetcher** [@storeorderedstreamer]: The store-ordered streamer prefetcher records store streams generated by producers (e.g., stores to memory). When a consumer (e.g., a load) accesses a line that is part of a producer's store sequence, the prefetcher triggers a series of prefetches for subsequent cache lines in the producer’s sequence. This effectively streams data from a producer to a consumer even when their access orders differ.

- **SMS Prefetcher** [@sms]: The Spatial Memory Streaming (SMS) prefetcher tracks access patterns in memory regions using two main components: an accumulation table that records access history for each region, and a pattern history table that stores spatial signatures. When a memory region is accessed, its pattern signature is matched against stored patterns to prefetch future accesses within that region, improving spatial locality.

- **EBCP Prefetcher** [@ebcp]: The Epoch-Based Correlation (EBCP) prefetcher partitions the memory access stream into epochs, with boundaries defined by quiescence periods (idle times). It learns correlations between the first miss of an epoch and subsequent misses within that same epoch. Upon detection of a similar epoch trigger, the prefetcher replays the learned miss sequence to prefetch data ahead of demand.

- **Feedback-Directed Prefetcher** [@feedbackdirected]: The feedback-directed prefetcher monitors feedback metrics such as prefetch accuracy, lateness, and cache pollution. It dynamically adjusts prefetch parameters like prefetch degree based on these metrics over sampling intervals. This tuning process helps balance the trade-off between cache performance improvement and bandwidth usage.

- **Temporal Memory Streaming Prefetcher** [@temporalmemorystreaming]: The temporal memory streaming prefetcher records instruction access sequences in a circular miss-order buffer. When an instruction is fetched, it checks the circular buffer for any matching address, and if a match is found, it prefetches subsequent instructions from the recorded sequence. A directory maps instruction addresses to sequence positions, allowing the prefetcher to maintain and replay instruction fetch sequences efficiently.

- **Linear Prefetcher** [@linear]: The linear prefetcher improves correlated prefetching by transforming irregular memory access patterns through correlation-based address linearization. By mapping related non-contiguous addresses into a logically linear sequence, the prefetcher enables more effective prediction of future accesses without assuming inherent spatial locality.

- **B-Fetch Prefetcher** [@bfetch]: The B-fetch prefetcher uses branch prediction information to guide instruction prefetching in chip multiprocessors. By following predicted branch outcomes and targets, it prefetches instructions along likely future control-flow paths rather than relying only on past fetch history. This approach improves prefetch accuracy and timeliness for workloads with irregular control flow.

- **IMP Prefetcher** [@indirectmemory]: The indirect memory prefetcher detects indirect memory access patterns where future addresses depend on previous load values (e.g., pointer-based or indexed accesses). By learning the correlation between index values and their corresponding target addresses, the prefetcher predicts future accesses based on the learned index-to-target relationship.

- **Best-Offset Hardware Prefetcher** [@bestoffset]: The best-offset prefetcher evaluates a predefined set of candidate address offsets and tracks the usefulness of each in generating prefetches. The prefetcher selects the highest-scoring offset based on periodic evaluation and continuously adapts to changes in program access behavior by re-evaluating the offsets.

- **F-TDC Prefetcher** [@f_tdc_prefetcher]: The F-TDC prefetcher learns spatial access footprints at the page level using compact footprint vectors. It predicts future memory accesses within a page by proactively fetching cache blocks associated with a previously accessed page, which is particularly useful in tagless DRAM cache designs.

- **Graph Prefetcher** [@graph]: The graph prefetcher utilizes programmer- or compiler-provided graph traversal semantics to guide prefetching. It identifies memory access patterns corresponding to graph traversal operations and prefetches connected nodes ahead of traversal, optimizing access to irregular graph data structures.

- **TEMPO Prefetcher** [@tempo]: The TEMPO prefetcher leverages address translation activity as a trigger for prefetching. It combines physical page mappings with cache-line offsets to predict future memory accesses. Non-speculative prefetches are issued as part of the address translation process, allowing better overlap between memory latency and translation.

- **Domino Temporal Data Prefetcher** [@domino]:The domino prefetcher uses two history tables to capture first-order and second-order correlations between cache misses. Predictions from these tables are chained together, providing more accurate prefetching by capturing deeper temporal relationships. This technique builds on the idea of "domino" effects, where early misses influence later prefetches.

- **Event-Triggered Programmable Prefetcher** [@eventtriggered]: The event-triggered prefetcher is programmable, where prefetching occurs only when specific events (defined by the programmer or system) are triggered based on runtime memory access characteristics (e.g., stride, latency, timing). This allows the prefetching strategy to be adapted dynamically to the workload characteristics.

- **LearnPrefetch Prefetcher** [@hashemi2018]: The Learnprefetch prefetcher uses an LSTM model to learn long-range temporal dependencies from address delta sequences derived from cache misses. By training on historical memory miss sequences, the LSTM predicts future address deltas and generates prefetch candidates, capturing complex temporal access patterns.

- **LearnCluster Prefetcher** [@hashemi2018]: An extension of the Learnprefetch, the Learncluster prefetcher clusters memory accesses using k-means clustering. Each cluster learns its own set of address deltas using the shared LSTM encoder. This improves prediction accuracy by specializing in different access patterns across clusters.

- **Bingo Spatial Data Prefetcher** [@bingo]: The bingo prefetcher tracks memory access patterns within regions using bit-vector footprints. It associates these footprints with specific trigger events. When a trigger matches a stored footprint, prefetches are issued for all cache blocks within the region indicated by the matched footprint.

- **DSPatch Prefetcher** [@dspatch]: The Dual Spatial Pattern (DSPatch) prefetcher maintains two types of footprints: accuracy-biased and coverage-biased patterns for each spatial signature. Depending on the access behavior, it selects the appropriate pattern to balance accuracy and coverage. Prefetches are issued by aligning the selected footprint with the region’s trigger offset.


- **Metadata Prefetcher** [@metadata]: The metadata prefetcher maintains per-PC metadata to track correlations between trigger addresses and their predicted target addresses. Confidence counters are associated with each correlation, and prefetches are only issued when these counters exceed a defined threshold, reducing the overhead of ineffective prefetches.

- **Perceptron Prefetcher** [@perceptron]: The Perceptron prefetcher uses a Perceptron classifier to filter prefetch candidates by evaluating features extracted from the memory access stream. The Perceptron is trained online using feedback from previous prefetch decisions, reinforcing high-confidence prefetches and suppressing low-confidence ones to reduce ineffective prefetches and cache pollution.

- **Triage Prefetcher** [@triage]: The triage prefetcher operates with on-chip metadata that tracks temporal correlations between consecutive memory addresses. The metadata cache is adaptively sized based on observed prefetch effectiveness, and prefetches are issued when high-confidence temporal correlations are identified.

- **IPCP Prefetcher** [@ipcp]: The IPCP prefetcher classifies memory access patterns based on PCs into constant stride, spatial signatures, or complex patterns. It uses specialized sub-prefetchers for each classification, allowing the prefetching strategy to adapt to different types of memory access patterns.

- **Neural Prefetcher** [@neural]: The neural prefetcher combines a global LSTM model for capturing long-range memory access patterns with per-PC linear models to learn localized instruction-specific behaviors. This hierarchical structure enables the prefetcher to adapt to both global trends and local, per-instruction access patterns.

- **Triangel Prefetcher** [@triangel]: The triangel prefetcher uses both pattern-based prediction and neighbor-based prediction. It identifies recurring access patterns through training and tracks Markov-style temporal metadata to predict neighboring address relationships. This hybrid approach allows the prefetcher to optimize both accuracy and timeliness in temporal prefetching.

- **Reference Prediction Table Prefetcher** [@referencepredictiontable]: The Reference Prediction Table (RPT) prefetcher records, for every load PC, the last address it referenced and the stride between its two most recent references. A per-entry finite state machine (initial, transient, steady, and unstable) tracks how consistently that stride repeats, and prefetches for the next address in the sequence are issued only once the entry reaches a state where the stride is considered reliable. This filters out loads whose strides fluctuate while still following regular array-like traversals closely.

- **SPPAM Prefetcher** [@spp; @ampm]: The Signature Pattern Prediction and Access Map (SPPAM) prefetcher combines two complementary mechanisms. Following the signature path prefetcher, the recent delta history of a page is folded into a compact signature that indexes a pattern table of delta candidates with confidence counters; the table is chased forward to predict a run of future deltas, with the length of that run throttled by the confidence accumulated along the path. In parallel, and following access map pattern matching, a per-region bitmap of previously touched block offsets is used to extrapolate the stride between recent accesses in the region independently of the PC, which lets the prefetcher keep issuing useful requests for regions whose delta signature has not yet been trained. This design follows a proposal submitted to the 4th Data Prefetching Championship, co-located with HPCA 2026.

- **Forest Prefetcher** [@forest]: The forest prefetcher targets GPU unified virtual memory, where a tree-based neighboring prefetcher covers each memory region with a full binary tree of fixed-size leaves: a fault on any page in a leaf migrates that whole leaf, and once more than half of a node's descendant pages are resident the remainder of that node is migrated as well, with the check cascading up the tree. Because a single fixed tree shape suits neither streaming nor sparse access patterns, Forest samples each memory object's early accesses, classifies its pattern as linear/streaming or as non-linear with high-coverage high-intensity, high-coverage low-intensity, or low-coverage behavior, and reconfigures that object's tree size and leaf size to the shape that best fits the observed pattern.

- **Pythia Prefetcher** [@pythia]: The Pythia prefetcher formulates prefetching as a reinforcement-learning problem. Each demand request forms a state built from program context—by default the program counter combined with the current delta, together with the sequence of the last four deltas—from which the prefetcher selects an action out of a fixed list of prefetch offsets. Every action taken is held in an evaluation queue until a later demand request either uses its prefetch or the entry ages out, at which point the action receives a reward that grades prefetch accuracy and timeliness while also accounting for memory bandwidth usage. Q-values are stored per feature and tile-coded across several hash planes, and are updated online with SARSA under an epsilon-greedy policy, so the prefetcher adapts its offset selection to program phases without offline training.

# References

- [@markovpredictor]
- [@correlation]
- [@hds]
- [@tcp]
- [@ghb]
- [@storeorderedstreamer]
- [@sms]
- [@ebcp]
- [@feedbackdirected]
- [@temporalmemorystreaming]
- [@linear]
- [@bfetch]
- [@indirectmemory]
- [@bestoffset]
- [@f_tdc_prefetcher]
- [@graph]
- [@tempo]
- [@domino]
- [@eventtriggered]
- [@hashemi2018]
- [@bingo]
- [@dspatch]
- [@metadata]
- [@perceptron]
- [@triage]
- [@ipcp]
- [@neural]
- [@triangel]

[^1]: These authors contributed equally to this work.