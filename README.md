

Reader Luninaries
=================

The is a Web-based application used to interactively read a set of three books:

  1. The Works of Horace
  2. Essays by Francis Bacon
  3. The Prince by machiavelli

Through the use of this tool the active reader (you) will enhance their use and understanding of the corpus. The whole thing kinda, sorta works like a back-of-the-book index but on steroids.


Requirements
------------

Reader Luminaries ought to run on any computer with Python version 3.12 or greater installed. 


Installation
------------

First, Reader Luminaries is a Python application. Open your terminal and run the following command, and version number greater than or equal to 3.12 ought to work:

    python --version

Second, Reader Luminaries requires the installation of [Ollama](https://ollama.com), a tool making it easy to run generative-AI applications on your local computer. Visit https://ollama.com/download and install Ollama. It is not hard. I promise.

Third, Reader Luminaries is configured to use two specific large language mnodels. Open your terminal and install LLama2:

    ollama pull deepseek-v3.1:671b-cloud

Then install nomic-embed-text:

    ollama pull nomic-embed-text:latest

Fourth, as if this writing, Reader Luminaries can only be downloaded from GitHub. Open your terminal and run the following command which will download the Reader Lite software. Mind you, since the application includes a number of indexes, the download is not small, but it is not too big either:

    git clone https://github.com/ericleasemorgan/reader-luminaries.git

Fifth, install the software, and begin by changing directories to where the software was downloaded:

    cd reader-luminaries

Use pip to do the actual installation, and power-users may want to install the tool in a virtual environment:

    pip install .

If you got this far, then the hard parts are complete.

Sixth, launch Reader Lite with the following command:

    flask --app reader_luminaries run --debug

Finally, open http://127.0.0.1:5000 in your Web browser, and you ought to see something very similar to the following:


<img width="600" height="349" alt="screenshot" src="https://github.com/user-attachments/assets/66b5ab89-1718-4a09-b2e9-0b12574e0989" />


Congratuations, you have successfully installed and launched Reader Luminaries. Whew.

Next time, just run the following command to pick up where you left off:

    flask --app reader_luminaries run --debug

While I can write rubust Python applications, I am still rusty on the writing of Python installation tools. Any help with the above instructions would be greatly appreciated!

---
Eric Lease Morgan &lt;eric_morgan@infomotions.com&gt;  
October 20, 2025