# How to setup DEMO

Apache has its own user `www-data`.
In the folder of the project we may create a `venv`
```python
#!/bin/sh
su -s /bin/bash www-data
```

Once loggined in, in `demo` dir on server we can install
ARElight:
```python
pip install git+https://github.com/nicolay-r/arelight@v0.24.0
```

Once installed, we may consider python scripts 
```python
#!/var/www/demo/venv/bin/python3
```

So that we have access to the internal venv.

For the manual environment variables, need to use (https://httpd.apache.org/docs/2.4/mod/mod_env.html#setenv)