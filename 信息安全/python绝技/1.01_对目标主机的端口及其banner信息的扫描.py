import socket


def retBanner(ip, port):
    try:
        socket.setdefaulttimeout(10)
        s = socket.socket()
        s.connect((ip, port))
        banner = s.recv(1024)
        return banner
    except:
        print('Time out')


def checkVulns(banner):
    if 'vsFTPd' in banner:
        print('[+] vsFTPd is vulnerable.')
    elif 'FreeFloat Ftp Server' in banner:
        print('[+] FreeFloat Ftp Server is vulnerable.')
    else:
        print('[-] FTP Server is not vulnerable.')
    return


def main():
    ips = ['10.10.10.128', '10.10.10.160']
    port = 21
    banner1 = retBanner(ips[0], port)
    if banner1:
        print('[+] ' + ips[0] + ": " + banner1.strip('\n'))
        checkVulns(banner1)
    banner2 = retBanner(ips[1], port)
    if banner2:
        print('[+] ' + ips[1] + ": " + banner2.strip('\n'))
        checkVulns(banner2)


if __name__ == '__main__':
    main()
