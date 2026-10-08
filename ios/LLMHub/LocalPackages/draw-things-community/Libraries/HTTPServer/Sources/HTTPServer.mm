#import "httplib.h"
#import "HTTPServer.h"
#import <iostream>

@implementation HTTPServerResponse

- (instancetype) initWithBody:(NSData *)body statusCode:(NSInteger)statusCode MIMEType:(NSString *)MIMEType
{
  if (self = [super init]) {
    _body = body;
    _statusCode = statusCode;
    _MIMEType = MIMEType;
    _requestHeaders = [NSMutableDictionary new];
  }
  return self;
}

@end

@implementation HTTPServer {
  httplib::Server *_server;
}

- (instancetype)init
{
  self = [super init];
  if (self) {
    _server = new httplib::Server();
    _server->set_logger([](const httplib::Request &request, const httplib::Response &response) {
      std::cout << "Processed request " << request.method << " " << request.path << "\n";
    });
  }
  return self;
}

- (void)addPOSTRouteWithPath:(const char * _Nonnull)path callback:(nonnull Callback)callback
{
  _server->Post(path, [callback](const httplib::Request &request, httplib::Response &response) {
    NSData *data = [NSData dataWithBytes:request.body.data() length:request.body.size()];
    HTTPServerResponse *serverResponse = callback(data);
    response.status = (int)serverResponse.statusCode;
    [serverResponse.requestHeaders enumerateKeysAndObjectsUsingBlock:^(NSString *key, NSString *value, BOOL *stop) {
      response.set_header(std::string(key.UTF8String), std::string(value.UTF8String));
    }];
    std::string content_type = serverResponse.MIMEType != nil ? std::string(serverResponse.MIMEType.UTF8String) : std::string("text/plain");
    response.set_content((const char *)serverResponse.body.bytes, serverResponse.body.length, content_type);
  });
}

- (void)addGETRouteWithPath:(const char * _Nonnull)path callback:(nonnull Callback)callback
{
  _server->Get(path, [callback](const httplib::Request &request, httplib::Response &response) {
    NSData *data = [NSData dataWithBytes:request.body.data() length:request.body.size()];
    HTTPServerResponse *serverResponse = callback(data);
    response.status = (int)serverResponse.statusCode;
    std::string content_type = serverResponse.MIMEType != nil ? std::string(serverResponse.MIMEType.UTF8String) : std::string("text/plain");
    response.set_content((const char *)serverResponse.body.bytes, serverResponse.body.length, content_type);
  });
}

- (BOOL)bindToIP:(const char * _Nonnull)ip port:(int)port
{
  return _server->bind_to_port(ip, port);
}

- (void)listenAfterBinding
{
  _server->listen_after_bind();
}

- (void)stop
{
  _server->stop();
}

- (void)dealloc
{
  _server->stop();
  delete _server;
}

@end
