#import <stdlib.h>
#import <Foundation/Foundation.h>

NS_ASSUME_NONNULL_BEGIN

#ifdef __cplusplus
extern "C" {
#endif

@interface HTTPServerResponse : NSObject

@property (nonatomic, strong) NSData *body;
@property (nonatomic, copy) NSString *MIMEType;
@property (nonatomic) NSInteger statusCode;
@property (nonatomic, strong) NSMutableDictionary<NSString *, NSString*> *requestHeaders;

- (instancetype) initWithBody:(NSData *)data statusCode:(NSInteger)statusCode MIMEType:(NSString *)MIMEType;

@end

typedef HTTPServerResponse *_Nonnull(^Callback)(NSData *);

@interface HTTPServer : NSObject

- (void)addPOSTRouteWithPath:(const char * _Nonnull)path callback:(Callback)callback;
- (void)addGETRouteWithPath:(const char * _Nonnull)path callback:(nonnull Callback)callback;
- (BOOL)bindToIP:(const char * _Nonnull)ip port:(int)port;
- (void)listenAfterBinding;
- (void)stop;

@end



#ifdef __cplusplus
}
#endif

NS_ASSUME_NONNULL_END
