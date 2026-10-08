"""Entry point for the packaged engine (fiducia-engine.exe).

In a frozen build, every worker process the orthorectification pool starts
re-runs this executable. freeze_support() must come first so those workers
go straight to their task instead of importing the web server and starting a
second copy of the engine.
"""

import multiprocessing

if __name__ == "__main__":
    multiprocessing.freeze_support()

    import server

    server.main()
