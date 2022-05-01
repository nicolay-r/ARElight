FROM ubuntu:18.04

# Requirements for python installation, resource downloading tools, etc.
RUN apt-get update && apt-get install -y git curl wget zlib1g-dev libssl-dev \
                                         build-essential libsqlite3-dev \
                                         libicu-dev locales libbz2-dev zip

# Python 3.6
RUN curl -L https://raw.githubusercontent.com/yyuu/pyenv-installer/master/bin/pyenv-installer | bash
ENV PYENV_ROOT /root/.pyenv
ENV PATH /root/.pyenv/shims:/root/.pyenv/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
RUN pyenv install 3.6.2
RUN pyenv global 3.6.2

# Locales.
RUN sed -i -e 's/# en_US.UTF-8 UTF-8/en_US.UTF-8 UTF-8/' /etc/locale.gen && \
        dpkg-reconfigure --frontend=noninteractive locales

# Update pip.
RUN pip install --upgrade pip

# Inistall ARElight project.
COPY arelight /arelight
RUN ls -la
RUN pip install /arelight/. -r /arelight/dependencies.txt

# Download DeepPavlov resources in advance.
COPY deeppavlov_data.sh /deeppavlov_data.sh
RUN chmod +x deeppavlov_data.sh
RUN /deeppavlov_data.sh

# Download required resources.
RUN python /arelight/download.py

EXPOSE 80

# Setup apache configs.
RUN apt-get clean && apt-get update && apt-get install -y apache2 python unzip
RUN a2enmod cgi
RUN echo export PATH=/root/.pyenv/shims:/root/.pyenv/bin:$PATH >> /etc/apache2/envvars

COPY apache_configs/* /etc/apache2/sites-enabled/

# Copy apache-based arelight demo.
COPY demo /var/www/demo
RUN chmod +x /var/www/demo/wui_bert.py
RUN chmod +x /var/www/demo/wui_nn.py

# Mystem3 fix
RUN git clone https://github.com/nlpub/pymystem3.git
RUN cp /pymystem3/pymystem3/mystem.py/ /root/.pyenv/versions/3.6.2/lib/python3.6/site-packages/pymystem3/

# Copy BRAT toolkit.
COPY brat /var/www/brat

RUN chmod 777 -R /root/.pyenv && chmod 777 /root

# Setup entrypoint
COPY entrypoint.sh /entrypoint.sh

# Modify a home folder to the demo.
RUN usermod -d /var/www/demo www-data
# Granting www-data permissions in demo.
RUN chown -R www-data:www-data /var/www/demo

RUN chmod +x entrypoint.sh
ENTRYPOINT ["/entrypoint.sh"]
CMD ["apache2ctl", "-D", "FOREGROUND"]
