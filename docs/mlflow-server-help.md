mlflow server --help       
Usage: mlflow server [OPTIONS]

  Run the MLflow tracking server.

  The server listens on http://localhost:5000 by default and only accepts      
  connections from the local machine. To let the server accept connections     
  from other machines, you will need to pass ``--host 0.0.0.0`` to listen on   
  all network interfaces (or a specific interface address).

Options:
  --backend-store-uri PATH        URI to which to persist experiment and run   
                                  data. Acceptable URIs are SQLAlchemy-        
                                  compatible database connection strings       
                                  (e.g. 'sqlite:///path/to/file.db') or local  
                                  filesystem URIs (e.g.
                                  'file:///absolute/path/to/directory'). By    
                                  default, data will be logged to the
                                  ./mlruns directory.
  --registry-store-uri URI        URI to which to persist registered models.   
                                  Acceptable URIs are SQLAlchemy-compatible    
                                  database connection strings (e.g.
                                  'sqlite:///path/to/file.db'). If not
                                  specified, `backend-store-uri` is used.      
  --default-artifact-root URI     Directory in which to store artifacts for    
                                  any new experiments created. For tracking    
                                  server backends that rely on SQL, this       
                                  option is required in order to store
                                  artifacts. Note that this flag does not      
                                  impact already-created experiments with any  
                                  previous configuration of an MLflow server   
                                  instance. By default, data will be logged    
                                  to the mlflow-artifacts:/ uri proxy if the   
                                  --serve-artifacts option is enabled.
                                  Otherwise, the default location will be      
                                  ./mlruns.
  --serve-artifacts / --no-serve-artifacts
                                  Enables serving of artifact uploads,
                                  downloads, and list requests by routing      
                                  these requests to the storage location that  
                                  is specified by '--artifacts-destination'    
                                  directly through a proxy. The default        
                                  location that these requests are served      
                                  from is a local './mlartifacts' directory    
                                  which can be overridden via the '--
                                  artifacts-destination' argument. To disable  
                                  artifact serving, specify `--no-serve-       
                                  artifacts`. Default: True
  --artifacts-only                If specified, configures the mlflow server   
                                  to be used only for proxied artifact
                                  serving. With this mode enabled,
                                  functionality of the mlflow tracking
                                  service (e.g. run creation, metric logging,  
                                  and parameter logging) is disabled. The      
                                  server will only expose endpoints for        
                                  uploading, downloading, and listing
                                  artifacts. Default: False
  --artifacts-destination URI     The base artifact location from which to     
                                  resolve artifact upload/download/list        
                                  requests (e.g. 's3://my-bucket'). Defaults   
                                  to a local './mlartifacts' directory. This   
                                  option only applies when the tracking        
                                  server is configured to stream artifacts     
                                  and the experiment's artifact root location  
                                  is http or mlflow-artifacts URI.
  -h, --host HOST                 The network address to listen on (default:   
                                  127.0.0.1). Use 0.0.0.0 to bind to all       
                                  addresses if you want to access the
                                  tracking server from other machines.
  -p, --port INTEGER              The port to listen on (default: 5000).       
  -w, --workers TEXT              Number of gunicorn worker processes to       
                                  handle requests (default: 1).
  --static-prefix TEXT            A prefix which will be prepended to the      
                                  path of all static paths.
  --gunicorn-opts TEXT            Additional command line options forwarded    
                                  to gunicorn processes.
  --waitress-opts TEXT            Additional command line options for
                                  waitress-serve.
  --expose-prometheus TEXT        Path to the directory where metrics will be  
                                  stored. If the directory doesn't exist, it   
                                  will be created. Activate prometheus
                                  exporter to expose metrics on /metrics       
                                  endpoint.
  --app-name [basic-auth|basic-auth]
                                  Application name to be used for the
                                  tracking server. If not specified,
                                  'mlflow.server:app' will be used.
  --dev                           If enabled, run the server with debug        
                                  logging and auto-reload. Should only be      
                                  used for development purposes. Cannot be     
                                  used with '--gunicorn-opts'. Unsupported on  
                                  Windows.
  --help                          Show this message and exit.