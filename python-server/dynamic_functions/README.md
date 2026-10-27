# Dynamic Functions

Chat comes pre-installed here, along with Home, Bot and Setup. These bundled apps
are tracked in the Atlantis MCP server repository. Demo is created on first run.

Add your own tools in a separate app folder. To keep a custom app in its own
repository, clone it directly into this directory:

    cd python-server/dynamic_functions
    git clone git@gitlab.com:you/my-atlantis-app.git MyApp

Keep the bundled app folders in place; do not replace the entire directory.
Custom apps and runtime data are ignored by Git by default.

See ../README.dynamic_functions.md for authoring details.
