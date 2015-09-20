import pafy

video = pafy.new('https://www.youtube.com/watch?v=aaFZsc6Kgxo')

plurl = 'https://www.youtube.com/playlist?list=PLwlF1uffYtXqvJGMOkOqIFqgfvvc1YFnf'
playlist = pafy.get_playlist(plurl)

print playlist['items']
