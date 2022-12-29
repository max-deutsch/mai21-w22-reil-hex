
import paramiko
from scp import SCPClient
import getpass

def createSSHClient(server, port, user, password):
    client = paramiko.SSHClient()
    client.load_system_host_keys()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(server, port, user, password)
    return client

user_name=input('Enter user name:\n')
ssh = createSSHClient('ctld-n1.cs.technikum-wien.at', 22, user_name, getpass.getpass(prompt='Password: '))

def uploadGameFiles():
    path = '/home/' + user_name + '/hex'
    scp = SCPClient(ssh.get_transport())
    scp.put('hex_engine.py', remote_path=path)
    scp.put('CNN.py', remote_path=path)
    scp.put('MonteCarloTreeSearch.py', remote_path=path)
    scp.put('config.py', remote_path=path)
    scp.put('train.py', remote_path=path)
    scp.put('worker.py', remote_path=path)
    scp.put('run_training.sh', remote_path=path)
    scp.put('run_worker.sh', remote_path=path)
    return


def downloadNnet(remote_path, local_path):
    scp = SCPClient(ssh.get_transport())
    scp.get(remote_path, local_path)

def ssh_tests():
    stdin, stdout, stderr = ssh.exec_command('sinfo')
    result = str(stdout.read())
    print(result.replace('\\n', '\n'))


uploadGameFiles()

# downloadNnet('/home/ai21m012/hex/models/champion.pt', 'models_saved/champion.pt')
# ssh_tests()
