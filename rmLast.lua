-- Handy script to delete last model used
require 'functions'

if file_exists(LAST_MODEL_FILE) then
   f = io.open(LAST_MODEL_FILE,'r')
   path = f:read()
   modelString = f:read()
   print('MODEL USED : '..modelString)
   f:close()
else
   error(LAST_MODEL_FILE.." should exist")
end

print("Do you really want to delete last model ? Enter if okay Ctrl-C otherwise")
io.read()

os.execute("rm -r "..path)
print("Deleted last model successfully")
