package sk.ainet.lang.memory.plan

/**
 * Whether this build runs on a web host — Kotlin/JS or Kotlin/Wasm in a browser, a worker or
 * Node — where there is no file system to map weights from, a single array holds 2 GB and the
 * memory a tab can count on is what [PlannerProfile.BROWSER] assumes. Decides
 * [PlannerProfile.forDevice]; a compile-time fact per target, not a runtime probe.
 */
internal expect val isWebHost: Boolean
