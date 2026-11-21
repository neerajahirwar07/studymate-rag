# CPU Scheduling

The CPU scheduler decides which ready process runs next. First Come First Served runs processes in arrival order and can suffer from the convoy effect, where short jobs wait behind a long one.

Shortest Job First gives the minimum average waiting time but needs to know burst lengths in advance and can starve long jobs. Round Robin gives each process a fixed time quantum and then moves it to the back of the queue, which gives good response time. If the quantum is too large it behaves like FCFS, and if it is too small the context switch overhead dominates.

Priority scheduling runs the highest priority process first. Starvation of low priority processes is solved by aging, which gradually raises the priority of waiting processes.
